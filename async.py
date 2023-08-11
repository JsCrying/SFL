import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'
import sys
from path import Path
import numpy as np
import datetime
import torch
from torch import nn
from pandas import DataFrame
import pandas as pd

from utils_dataset import *
from utils_model import *
from utils_general import *
from utils_args import args_parser
from clients_side import *
from server_side import *
import math

  
from tensorboardX import SummaryWriter

#---------------------------------------------
#args.group == 2 即切两种：2层指的是第1层+第4层，
##TODO:聚合方案（1） ：相同大小的聚合，不同大小的不聚合，取ACC平均值
#聚合方案（2）：第一层不聚合，其他聚合，测整体ACC
#args.group == 1 全部切4层
#---------------------------------------------
#%%----- main process ----
def SFL_over_SA(rule_iid ,K, Group):
    #------Random seed-----
    random_seed()
    args = args_parser()
    args.K = K
    args.Group = Group

    #%%-----------------------------------
    # The Core Process
    #-------------------------------------
    #TODO: 卫星时间
    """
    1.read SA-GS connect data from csv total five days
    # """
    data_csv = pd.read_csv('./20SA_60DAY_endtime.csv')#按照EndTime排序
    # data_csv = pd.read_csv('./sort_EndTime_SH_550km_5PLAN_20SAT_60days.csv')
    data_csv = np.array(data_csv)
    # data_csv = np.array(data_csv[0:40])#实验中只取400
    user_list_raw = data_csv[:,0]
    user_list = []
    for i in user_list_raw:
        j = i.split(sep='_')
        k = int(j[1])
        user_list.append(k-1)

    endtime_list = data_csv[:,4]#按照Endtime的时间
    time_start = data_csv[:,3][0]#Startime的时间，先走先通信
    time_start = datetime.datetime.strptime(time_start,'%Y-%m-%d %H:%M:%S')
    data_times =[datetime.datetime.strptime(data_str,'%Y-%m-%d %H:%M:%S') for data_str in endtime_list]
    data_times = [((data_time - time_start).days*24)+((data_time - time_start).seconds/3600) for data_time in data_times] #最多10天即240H


    #%%=====================================
    program = "SFL AlexNet"
    print(f"---------{program}----------")              
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #----- data asign -------------------
    rule_iid = rule_iid   
    data_path = 'Folder/'
    # data_obj = DatasetObject(dataset='mnist', n_client = args.num_users , seed=23, rule='iid', rule_arg = 0.6, unbalanced_sgm=0, data_path=data_path)
    data_obj = DatasetObject(dataset=args.dataset, n_client = args.num_users , seed=23, rule= rule_iid, rule_arg = 0.6, unbalanced_sgm=0, data_path=data_path)
    
    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y
    cent_x = np.concatenate(clnt_x, axis = 0)#训练集
    cent_y = np.concatenate(clnt_y, axis = 0)
    user_id = list(set(user_list))

    #%%---- Tensorboard for checking results----------------------------------------
    suffix ='Async_' + str(args.Group) 
    suffix += '_SFL_' + str(args.num_users) +'_' +str(args.K) + '_' + str(args.dataset)
    suffix += '_LR%f_BS%d_E%d_' %(args.local_lr, args.local_bs, args.local_ep)
    data_path_tb = 'Folder/'
    if (not os.path.exists('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))):
        # os.mkdir('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
        os.makedirs('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
   
    saved_itr = -1 #用作结果记录
    writer = SummaryWriter('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
   
    #%%----- Split model Client side ----
    # if args.dataset == 'mnist':
    #     clnt_model_func = lambda: CNNmnist_client_side()
    if args.dataset =='Cifar10' or 'CIFAR10':
        clnt_model_func_G4 = lambda: AlexNet_client_side_C4()#第1-4层
        clnt_model_func_G2 = lambda: AlexNet_client_side_C2()#第3-4层
    
    torch.manual_seed(37)
    init_net_client_G4 = clnt_model_func_G4()
    load_pretrained(init_net_client_G4, {0: 0}, 'features')
    init_net_client_G2 = clnt_model_func_G2()
    load_pretrained(init_net_client_G2, {8: 3}, 'features')
    #TODO:check 两层网络是否参数和四层网络的后两层一致
    clnt_models = list(range(args.num_users))
    FL_model = clnt_model_func_G4() #取较大网络模型
    FL_model.load_state_dict(copy.deepcopy(dict(init_net_client_G4.named_parameters())))
    FL_G2 = clnt_model_func_G2()
    #----参数一致 ----------------------------------
    if Group == 2:
        keys_G4 = list(dict(init_net_client_G4.named_parameters()).keys())
        keys_G2 = list(dict(init_net_client_G2.named_parameters()).keys())
        init_G4 = copy.deepcopy(dict(init_net_client_G4.named_parameters()))
        init_G2 = copy.deepcopy(dict(init_net_client_G2.named_parameters()))
        init_G2[keys_G2[2]] = init_G4[keys_G4[6]]
        init_G2[keys_G2[3]] = init_G4[keys_G4[7]]
        FL_G2.load_state_dict(copy.deepcopy(init_G2))
    print('client model G4', init_net_client_G4)
    print('client model G2', init_net_client_G2)


    clnt_smashed = [[] for i in range(args.num_users)]
    clnt_glob_graditent = [[] for i in range(args.num_users)]
    AN_loss_train_list = [[] for i in range(args.num_users)]
    AN_acc_train_list = [[] for i in range(args.num_users)]
       
    #%%----- Auxiliary Network ----
    AN_model_func = lambda: ANet_C4()#所有的辅助网络都是一样的
    torch.manual_seed(37)
    init_local_AN = AN_model_func()
    AN_net = list(range(args.num_users))

    #%%----- Split model Server side----
    torch.manual_seed(37)
    net_server = AlexNet_server_side_C4()
    if args.Group == 1:
        load_pretrained(net_server, {3:0, 6:4, 8:6, 10:8}, 'features')
        load_pretrained(net_server, {1:1, 4:4}, 'classifiers')
    elif args.Group == 2:
        load_pretrained(net_server, {10:0}, 'features')
        load_pretrained(net_server, {1:1, 4:4}, 'classifiers')
    net_server.to(device)

    #---- Server side Arguments-----
    FL_loss = []
    FL_acc = []
    FL_loss_trn = []
    FL_acc_trn = []
    recv_iter_list = []
    w_locals_client = []

    #%%---- 方案1：所有用户都分割一样的4层网络------
    if args.Group == 1:
        for clnt in user_id:#初始轮训分发
            #----load the global federated model------
            clnt_models[clnt] =  clnt_model_func_G4().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(init_net_client_G4.named_parameters())))
            AN_net[clnt] = AN_model_func().to(device)
            AN_net[clnt].load_state_dict(copy.deepcopy(dict(init_local_AN.named_parameters())))
   
    #TODO：第一轮也有时间，但统一没有记录
    elif args.Group == 2:
    #---- 方案2：一半用户分割4层一半用户分割2层网络--
        m = max(int(args.frac * args.num_users), 1)
        idx_G4 = list((np.random.choice(user_id, m, replace = False)))
        # users_id = copy.deepcopy(user_list)
        idx_G2 = [id for id in user_id if (id not in idx_G4)]
        for clnt in idx_G4:
            clnt_models[clnt] =  clnt_model_func_G4().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(init_net_client_G4.named_parameters())))
            AN_net[clnt] = AN_model_func().to(device)
            AN_net[clnt].load_state_dict(copy.deepcopy(dict(init_local_AN.named_parameters())))            
        for clnt in idx_G2:
            clnt_models[clnt] =  clnt_model_func_G2().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(init_net_client_G2.named_parameters())))
            AN_net[clnt] = AN_model_func().to(device)
            AN_net[clnt].load_state_dict(copy.deepcopy(dict(init_local_AN.named_parameters())))             

    #----内存释放----------
    del init_net_client_G2
    del init_net_client_G4
    del init_local_AN

    #%%---从已经拿到初始网络之后开始训练程序—-----
    iter = 0        
    

    #%%Training test-------------------------------------------
    [loss_trn, acc_trn] = evaluate(net = copy.deepcopy(FL_model).to(device), 
        dataset_tst = cent_x, 
        dataset_tst_label = cent_y , net_server = copy.deepcopy(net_server), device = device)                 
    print("'Federated Iteration %3d','loss_train: %.4f', 'acc_trn: %.4f'" %( iter, loss_trn, acc_trn))         
    FL_acc_trn.append(acc_trn)
    FL_loss_trn.append(loss_trn)       

    writer.add_scalars('Accuracy/Train', { 'All Clients': FL_acc_trn[saved_itr]}, 0 )
    writer.add_scalars('Loss/Train', {'All Clients': FL_loss_trn[saved_itr]}, 0) 
    #%% Testing-------------------------------------------------
    [loss_tst, acc_tst] = evaluate(net = copy.deepcopy(FL_model).to(device), 
        dataset_tst = data_obj.test_x, 
        dataset_tst_label = data_obj.test_y , net_server = copy.deepcopy(net_server), device = device)                     
    print("'Federated Iteration %3d', 'loss_tst: %.4f', 'acc_tst: %.4f' "%(iter, loss_tst, acc_tst))      
    FL_acc.append(acc_tst)
    FL_loss.append(loss_tst)
    saved_itr +=1
    writer.add_scalars('Accuracy/Test',    {'All Clients': FL_acc[saved_itr]   }, 0)
    writer.add_scalars('Loss/Test', {'All Clients': FL_loss[saved_itr] }, 0 )

    server_iter = 0
    do_grad_top = args.num_users
    sats = [Client(args, clnt_models[clnt], clnt, device, AN_net[clnt],
                do_srv2clnt_grad = do_grad_top,
                dataset_train = clnt_x[clnt], dataset_test = None,
                idxs = clnt_y[clnt], idxs_test = None,
                dataset_name = args.dataset)
                for clnt in range(args.num_users)]

    #%%------按照星历表遍历----------------------------------------------------
    for idx, clnt in enumerate(user_list):    #按照星历表遍历       
        clnt_endtime = data_times[idx] #卫星离开的时间，星历表应该按照Endtime排序

        #----学习率衰减==============================
        # if (idx+1) % 20 == 0 :
        #     args.local_lr = args.local_lr * 0.9
        #     args.local_lr = max(1e-3,args.local_lr)
        # if (idx+1) % args.num_users == 0: 
        #     args.local_lr = args.local_lr * (0.998 ** idx)   
        #     args.local_lr = max(1e-3,args.local_lr) 
        # print(args.local_lr)
        #============================================
        clnt_models[clnt].train()
        AN_net[clnt].train()
        for params in clnt_models[clnt].parameters():
            params.requires_grad = True

        for params in AN_net[clnt].parameters():
            params.requires_grad = True

        sats[clnt] = Client(args, clnt_models[clnt], clnt, device, AN_net[clnt],
                        recv_iter = sats[clnt].recv_iter,
                        do_srv2clnt_grad = do_grad_top,
                        dataset_train = clnt_x[clnt], dataset_test = None, idxs = clnt_y[clnt],
                        idxs_test = None, dataset_name = args.dataset) #测试集没有放在client
        [w_client, AN_params, smashed_list, AN_loss_train, AN_acc_train] = sats[clnt].train(net = copy.deepcopy(clnt_models[clnt].to(device)))

        recv_iter_list.append(sats[clnt].recv_iter)
        w_locals_client.append(copy.deepcopy(w_client))
        AN_loss_train_list[clnt].append(AN_loss_train)
        AN_acc_train_list[clnt].append(AN_acc_train)
        AN_net[clnt].load_state_dict(copy.deepcopy(dict(AN_params)))           
        #Frezz model
        for params in AN_net[clnt].parameters():
            params.requires_grad = False

        # clnt_smashed[clnt] = copy.deepcopy(smashed_list)
        #TODO: 对smahed_data做改进
        #TODO: server_train
        #TODO：用户记录最后一个batch的smashed_data， labels                             
        #TODO: 10张10张输入而不是60*10张一起输入      
        for smashed_labels in smashed_list:#长度为64*32 即local_bs*local_ep
            smashed_data = smashed_labels[0]
            local_labels = smashed_labels[1]
            [net_server_update, global_gradient] = train_server(net_server, smashed_data, local_labels, device=device, lr=args.local_lr)
            clnt_glob_graditent[clnt] = global_gradient #最后一次的梯度
            net_server = copy.deepcopy(net_server_update) #有更新，但是很慢
        server_iter = server_iter + 1

        #TODO: 测试 FL_model + server_model               
        if len(w_locals_client) >= args.K:             #iter%2==0,5,10
            args.async_lr = 1 if args.K % args.num_users == 0 else 0.1
            w_old = copy.deepcopy(FL_model.to(device).state_dict()) 
            if args.Group == 2:
                FL_G4_params, FL_G2_params = FedBuff_del_inputlayer(w_old,w_locals_client, args.async_lr)
                 
            elif args.Group == 1:
                # FL_G4_params = FedBuff(w_old, w_locals_client, args.async_lr)
                FL_G4_params = FedAsyncPoly(w_old, w_locals_client, recv_iter_list, server_iter, args)
            #清除Buff-----------------------------
            recv_iter_list = []
            w_locals_client = []
            FL_model.load_state_dict(FL_G4_params) #只对客户端模型做FL
            if args.Group == 2 and FL_G2_params != 0:
                if FL_G2_params !=0:
                    FL_G2.load_state_dict(FL_G2_params)  
                else: FL_G2 = FL_G2
            # FL_AN.load_state_dict(AN_glob_client)
            #TODO:用FL_model做Client端？

            if (idx + 1) % (args.num_users) == 0:  # 衰减步长 200
                args.local_lr = args.local_lr * (0.992 ** (4.0 * pow((idx + 1) / (args.num_users), 1.0 / 8.0)))
                args.local_lr = max(1e-3, args.local_lr)
            print(args.local_lr)
            iter += 1

        else:
            FL_model = FL_model #
            FL_G2 = FL_G2

        #%%Training test-------------------------------------------
        [loss_trn, acc_trn] = evaluate(net = copy.deepcopy(FL_model).to(device), 
            dataset_tst = cent_x, 
            dataset_tst_label = cent_y , net_server = copy.deepcopy(net_server), device = device)                 
        print("'Federated Iteration %3d','loss_train: %.4f', 'acc_trn: %.4f'" %( iter, loss_trn, acc_trn))      
        FL_acc_trn.append(acc_trn)
        FL_loss_trn.append(loss_trn)       

        writer.add_scalars('Accuracy/Train', { 'All Clients': FL_acc_trn[saved_itr]}, clnt_endtime )
        writer.add_scalars('Loss/Train', {'All Clients': FL_loss_trn[saved_itr]}, clnt_endtime) 
        #%% Testing-------------------------------------------------
        [loss_tst, acc_tst] = evaluate(net = copy.deepcopy(FL_model).to(device), 
            dataset_tst = data_obj.test_x, 
            dataset_tst_label = data_obj.test_y , net_server = copy.deepcopy(net_server), device = device)                     
        print("'Federated Iteration %3d', 'loss_tst: %.4f', 'acc_tst: %.4f' "%(iter, loss_tst, acc_tst))      
        FL_acc.append(acc_tst)
        FL_loss.append(loss_tst)
        saved_itr +=1
        writer.add_scalars('Accuracy/Test',    {'All Clients': FL_acc[saved_itr]   },clnt_endtime)
        writer.add_scalars('Loss/Test', {'All Clients': FL_loss[saved_itr] }, clnt_endtime)
        writer.add_scalars('LR', {'All Clients': args.local_lr}, clnt_endtime)
        #%%---------------------------------回传-------------------------------------------------
        # Freeze model
        for params in FL_model.parameters():
            params.requires_grad = False
        for params in FL_G2.parameters():
            params.requires_grad = False

        if args.Group == 2:
            if clnt in idx_G4:
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_model.named_parameters())))

            elif clnt in idx_G2:
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_G2.named_parameters())))
        elif args.Group == 1:
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_model.named_parameters())))
        sats[clnt].recv_iter = server_iter

        if idx % 2000 == 0:
            record(FL_acc, FL_loss, FL_acc_trn, FL_loss_trn, args, rule_iid)

    record(FL_acc, FL_loss, FL_acc_trn, FL_loss_trn, args, rule_iid)
                # ==========================================
    print("Training and Evaluation completed!")

#%%--------Main-------------------------------------
#===================================================s

if __name__  == '__main__':
    # for Group_type in [2,1]:
    #     for rule_iid in ['iid', 'Noniid']:
    #         for K in [1]:
    #             SFL_over_SA(rule_iid, K, Group_type)
    #     for rule_iid in ['iid', 'Noniid']:
    #         for K in [5]:
    #             SFL_over_SA(rule_iid, K, Group_type)

    for Group_type in [1]:
        for rule_iid in ['iid']:
            for K in [1]:
                SFL_over_SA(rule_iid, K, Group_type)
        for rule_iid in ['iid']:
            for K in [10]:
                SFL_over_SA(rule_iid, K, Group_type)
    print('wait for check!')
