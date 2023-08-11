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
    """
    1.read SA-GS connect data from csv total five days
    # """
    data_csv = pd.read_csv('./20SA_60DAY_endtime.csv')#按照EndTime排序
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
    suffix = 'Async_' + str(args.Group) 
    suffix += '_SFL_' + str(args.num_users) + '_' + str(args.K) + '_' + str(args.dataset)
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
        clnt_model_same = lambda: VGG16_client_same()
        server_model_same = lambda: VGG16_server_same()

        clnt_model_diff_1 = lambda: VGG16_client_diff_1()
        clnt_model_diff_2 = lambda: VGG16_client_diff_2()
        server_model_diff = lambda: VGG16_server_diff()

    torch.manual_seed(37)

    init_net_client_same = []
    init_net_client_diff_1 = []
    init_net_client_diff_2 = []
    FL_model = []
    AN_model_func = []
    net_server = []

    clnt_models = list(range(args.num_users))
    AN_net = list(range(args.num_users))
    clnt_glob_graditent = [[] for i in range(args.num_users)]
    AN_loss_train_list = [[] for i in range(args.num_users)]
    AN_acc_train_list = [[] for i in range(args.num_users)]

    if args.Group == 1: # 同分割
        init_net_client_same = clnt_model_same()
        # pretrain
        load_pretrained(init_net_client_same, {x:x for x in [0, 2]}, 'features')

        FL_model = clnt_model_same()
        FL_model.load_state_dict(copy.deepcopy(dict(init_net_client_same.named_parameters())))

        AN_model_func = lambda: ANet_same()

        net_server = server_model_same()
        load_pretrained(net_server, {x:x-5 for x in [5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]}, 'features')
        load_pretrained(net_server, {x:x for x in [0, 3]}, 'classifier')
        # pretrain
        
    elif args.Group == 2: # 不同分割
        init_net_client_diff_1 = clnt_model_diff_1()
        # pretrain
        load_pretrained(init_net_client_diff_1, {x:x for x in [0, 2]}, 'features')
        init_net_client_diff_2 = clnt_model_diff_2()
        # pretrain
        load_pretrained(init_net_client_diff_2, {x:x for x in [0, 2, 5, 7]}, 'features')
        FL_diff_1 = clnt_model_diff_1()
        FL_diff_1.load_state_dict(copy.deepcopy(dict(init_net_client_diff_1.named_parameters())))

        FL_diff_2 = clnt_model_diff_2()
        FL_diff_2.load_state_dict(copy.deepcopy(dict(init_net_client_diff_2.named_parameters())))

        AN_model_func = lambda: ANet_diff()

        net_server = server_model_diff()
        # pretrain
        load_pretrained(net_server, {x:x-10 for x in [10, 12, 14, 17, 19, 21, 24, 26, 28]}, 'features')
        load_pretrained(net_server, {x:x for x in [0, 3]}, 'classifier')
    init_local_AN = AN_model_func()
    
    net_server.to(device)

    #---- Server side Arguments-----
    FL_loss = []
    FL_acc = []
    FL_loss_trn = []
    FL_acc_trn = []
    recv_iter_list = []
    w_locals_client = []
    idx_diff_1 = []
    idx_diff_2 = []

    #%%---- 方案1：所有用户都分割一样的1层网络------
    if args.Group == 1:
        for clnt in user_id: #初始轮训分发
            #----load the global federated model------
            clnt_models[clnt] =  clnt_model_same().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(init_net_client_same.named_parameters())))
            AN_net[clnt] = AN_model_func().to(device)
            AN_net[clnt].load_state_dict(copy.deepcopy(dict(init_local_AN.named_parameters())))
    elif args.Group == 2:
    #---- 方案2：一半用户分割1层一半用户分割2层网络--
        m = max(int(args.frac * args.num_users), 1)
        idx_diff_1 = list((np.random.choice(user_id, m, replace = False)))
        idx_diff_2 = [id for id in user_id if (id not in idx_diff_1)]
        for clnt in idx_diff_1:
            clnt_models[clnt] =  clnt_model_diff_1().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(init_net_client_diff_1.named_parameters())))
            AN_net[clnt] = AN_model_func().to(device)
            AN_net[clnt].load_state_dict(copy.deepcopy(dict(init_local_AN.named_parameters())))            
        for clnt in idx_diff_2:
            clnt_models[clnt] =  clnt_model_diff_2().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(init_net_client_diff_2.named_parameters())))
            AN_net[clnt] = AN_model_func().to(device)
            AN_net[clnt].load_state_dict(copy.deepcopy(dict(init_local_AN.named_parameters())))             

    #----内存释放----------
    del init_net_client_diff_1
    del init_net_client_diff_2
    del init_net_client_same
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

        for smashed_labels in smashed_list: # 长度为64*32 即local_bs*local_ep
            smashed_data = smashed_labels[0]
            local_labels = smashed_labels[1]
            [net_server_update, global_gradient] = train_server(net_server, smashed_data, local_labels, device=device, lr=args.local_lr)
            clnt_glob_graditent[clnt] = global_gradient #最后一次的梯度
            net_server = copy.deepcopy(net_server_update) #有更新，但是很慢
        server_iter = server_iter + 1

        # 测试 FL_model + server_model               
        if len(w_locals_client) >= args.K:             #iter%2==0,5,10
            args.async_lr = 1 if args.K % args.num_users == 0 else 0.1
            # TODO
            if args.Group == 1:
                w_old = copy.deepcopy(FL_model.to(device).state_dict()) 
                # FL_diff_1_params = FedBuff(w_old, w_locals_client, args.async_lr)
                FL_diff_params = FedAsyncPoly(w_old, w_locals_client, recv_iter_list, server_iter, args)
            elif args.Group == 2:
                # w_old_1 = copy.deepcopy(FL_diff_1.to(device).state_dict()) 
                w_old = copy.deepcopy(FL_diff_2.to(device).state_dict()) 
                FL_diff_1_params, FL_diff_2_params = FedBuff_del_inputlayer(w_old, w_locals_client, args.async_lr)
            #清除Buff-----------------------------
            recv_iter_list = []
            w_locals_client = []
            if args.Group == 1:
                FL_model.load_state_dict(FL_diff_params) #只对客户端模型做FL
            elif args.Group == 2 and FL_diff_2_params != 0:
                FL_diff_1.load_state_dict(FL_diff_1_params)
                FL_diff_2.load_state_dict(FL_diff_2_params)  

            if (idx + 1) % (args.num_users) == 0:  # 衰减步长 200
                args.local_lr = args.local_lr * (0.992 ** (4.0 * pow((idx + 1) / (args.num_users), 1.0 / 4.0)))
                args.local_lr = max(1e-3, args.local_lr)
            print(args.local_lr)
            iter += 1

        else:
            FL_model = FL_model
            FL_diff_1 = FL_diff_1
            FL_diff_2 = FL_diff_2

        #%%Training test-------------------------------------------
        [loss_trn, acc_trn] = evaluate(net = copy.deepcopy(FL_model).to(device), 
            dataset_tst = cent_x, 
            dataset_tst_label = cent_y , net_server = copy.deepcopy(net_server), device = device)                 
        print("'Federated Iteration %3d','loss_train: %.4f', 'acc_trn: %.4f'" % ( iter, loss_trn, acc_trn))      
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
        if args.Group == 1:
            # Freeze model
            for params in FL_model.parameters():
                params.requires_grad = False
            
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_model.named_parameters())))

        elif args.Group == 2:
            for params in FL_diff_1.parameters():
                params.requires_grad = False
            for params in FL_diff_2.parameters():
                params.requires_grad = False
            
            if clnt in idx_diff_1:
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_diff_1.named_parameters())))
            elif clnt in idx_diff_2:
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_diff_2.named_parameters())))
                
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
