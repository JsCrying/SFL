import os
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

  
from tensorboardX import SummaryWriter

#TODO:sync 同步需要user_id的识别
#TODO：卫星数据放在了数据分配后面，如果卫星部署改变了需要及时调整args.num_users
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
    #=====================================
    program = "SFL AlexNet on mnist"
    print(f"---------{program}----------")              
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #----- data asign -----
    rule_iid = rule_iid   
    data_path = 'Folder/'
    # data_obj = DatasetObject(dataset='mnist', n_client = args.num_users , seed=23, rule='iid', rule_arg = 0.6, unbalanced_sgm=0, data_path=data_path)
    data_obj = DatasetObject(dataset=args.dataset, n_client = args.num_users , seed=23, rule= rule_iid, rule_arg = 0.3, unbalanced_sgm=0, data_path=data_path)
    
    test_data = data_obj.test_x
    test_labels = data_obj.test_y

    #---- Tensorboard for checking results---
    suffix ='Sync_' + str(args.Group) 
    suffix += '_SFL_' + str(args.num_users) +'_' +str(args.K) + '_' + str(args.dataset)
    suffix += '_LR%f_BS%d_E%d_' %(args.local_lr, args.local_bs, args.local_ep)
    data_path_tb = 'Folder/'
    if (not os.path.exists('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))):
        # os.mkdir('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
        os.makedirs('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
   
    #----- Split model Client side ----
    # if args.dataset == 'mnist':
    #     clnt_model_func = lambda: CNNmnist_client_side()
    if args.dataset =='Cifar10' or 'CIFAR10':
        clnt_model_func_G4 = lambda: AlexNet_client_side_C4()#第1-4层
        clnt_model_func_G2 = lambda: AlexNet_client_side_C2()#第3-4层
    torch.manual_seed(37)
    init_net_client_G4 = clnt_model_func_G4()
    init_net_client_G2 = clnt_model_func_G2()
    #TODO:check 两层网络是否参数和四层网络的后两层一致
    clnt_models = list(range(args.num_users))
    FL_model = clnt_model_func_G4() #取较大网络模型
    FL_model.load_state_dict(copy.deepcopy(dict(init_net_client_G4.named_parameters())))
    FL_G2 = clnt_model_func_G2()
    #----参数一致 ----------------------------------
    keys_G4 = list(dict(init_net_client_G4.named_parameters()).keys())
    keys_G2 = list(dict(init_net_client_G2.named_parameters()).keys())
    init_G4 = copy.deepcopy(dict(init_net_client_G4.named_parameters()))
    init_G2 = copy.deepcopy(dict(init_net_client_G2.named_parameters()))
    # for j, k in enumerate (keys_G2):
    #     init_G2[k] = init_G4[keys_G4[j+4]]
    
    init_G2[keys_G2[2]] = init_G4[keys_G4[6]]
    init_G2[keys_G2[3]] = init_G4[keys_G4[7]]
    FL_G2.load_state_dict(copy.deepcopy(init_G2))
    # FL_G2.load_state_dict(copy.deepcopy(dict(init_net_client_G2.named_parameters())))
    print('client model G4', init_net_client_G4)
    print('client model G2', init_net_client_G2)

    clnt_smashed = [[] for i in range(args.num_users)]
    clnt_glob_graditent = [[] for i in range(args.num_users)]

    AN_loss_train_list = [[] for i in range(args.num_users)]
    AN_acc_train_list = [[] for i in range(args.num_users)]
       
    #----- Auxiliary Network ----
    AN_model_func = lambda: ANet_C4()#所有的辅助网络都是一样的
    torch.manual_seed(37)
    init_local_AN = AN_model_func()
    AN_net = list(range(args.num_users))
    # FL_AN = AN_model_func() #辅助网络可以不做FL
    # FL_AN.load_state_dict(copy.deepcopy(dict(init_local_AN.named_parameters())))


    #------FFC network-----
    torch.manual_seed(37)
    FFC_net = FFC_C2()
    FFC_net.to(device)
    FFC_net.train()

    #----- Split model Server side----
    torch.manual_seed(37)
    net_glob_server = AlexNet_server_side_C4()
    init_model_server = AlexNet_server_side_C4()
    if torch.cuda.device_count() > 1:
        print("We use",torch.cuda.device_count(), "GPUs")
        net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

    net_glob_server.to(device)
    print('Server model', net_glob_server)      
 
    #---- Initialization of net_model_server and net_server(server-side model)
    net_model_server = [net_glob_server for i in range(args.num_users)]
    net_server = copy.deepcopy(net_model_server[0]).to(device)

    #%%-----------------------------------
    # The Core Process
    #-------------------------------------
    #TODO: 卫星时间
    """
    1.read SA-GS connect data from csv total five days
    """
    data_csv = pd.read_csv('./sort_start_intvls_SH_550km_5PLAN_40SAT_20days.csv')
    data_csv = np.array(data_csv)
    # data_csv = np.array(data_csv[0:40])#实验中只取400
    user_list_raw = data_csv[:,0]
    user_list = []
    for i in user_list_raw:
        j = i.split(sep='_')
        k = int(j[1])
        user_list.append(k-1)
    # print (len(user_list))
    endtime_list = data_csv[:,4]
    time_start = data_csv[:,3][0]
    time_start = datetime.datetime.strptime(time_start,'%Y-%m-%d %H:%M:%S')
    data_times =[datetime.datetime.strptime(data_str,'%Y-%m-%d %H:%M:%S') for data_str in endtime_list]
    data_times = [((data_time - time_start).days*24)+((data_time - time_start).seconds/3600) for data_time in data_times] #最多10天即240H
    
    n_clnt = data_obj.n_client #args.num_users
    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y
    cent_x = np.concatenate(clnt_x, axis = 0)
    cent_y = np.concatenate(clnt_y, axis = 0)

    # user_list = [i for i in range(n_clnt)]#转成卫星
    train_acc_all = [[] for i in range(n_clnt)]
    train_loss_all = [[] for i in range(n_clnt)]
    user_id = list(set(user_list))
    #---- Server side Arguments-----
    FL_loss = []
    FL_acc = []
    FL_loss_trn = []
    FL_acc_trn = []

    w_locals_client = []
    # AN_locals_client = []


    #---- 方案1：所有用户都分割一样的4层网络------
    if args.Group == 1:
        for clnt in user_id:#初始轮训分发
            #----load the global federated model------
            clnt_models[clnt] =  clnt_model_func_G4().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_model.named_parameters())))
            AN_net[clnt] = AN_model_func().to(device)
            AN_net[clnt].load_state_dict(copy.deepcopy(dict(init_local_AN.named_parameters())))
    #TODO：第一轮也有时间
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


#------------------- Sync FL------------------
    s_list = [] #记录分发
    r_list = copy.deepcopy(user_id)   #记录待接收
    #如果没有新的更新就空转 ：防止不同的客户端训练频次不同，容易过拟合
    s_num = 0

    iter = 0
    # for iter in range(args.global_iters):        
    for idx, clnt in enumerate(user_list):    #按照星历表遍历        
        if clnt in s_list: #分发，注意到第一轮的分发可是统一初始化
            if args.Group == 2:
                if clnt in idx_G4:
                    clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_model.named_parameters())))
                    # AN_net[clnt].load_state_dict(copy.deepcopy(dict(FL_AN.named_parameters())))
                elif clnt in idx_G2:
                    clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_G2.named_parameters())))
                    # AN_net[clnt].load_state_dict(copy.deepcopy(dict(FL_AN.named_parameters())))             
            elif args.Group == 1:
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_model.named_parameters())))  
            
            s_list.remove(clnt)
            s_num += 1
            if (s_num > 0) and (s_num % n_clnt == 0):
                s_list = []
                r_list = copy.deepcopy(user_id)        

        elif clnt in r_list: #接收
            r_list.remove(clnt)
            clnt_models[clnt].train()
            AN_net[clnt].train()
            for params in clnt_models[clnt].parameters():
                params.requires_grad = True

            for params in AN_net[clnt].parameters():
                params.requires_grad = True

            need_ffc = True if args.Group == 2 and (clnt in idx_G2) else False
            local = Client(args, clnt_models[clnt], clnt, device, AN_net[clnt], dataset_train = clnt_x[clnt], dataset_test = None, idxs = clnt_y[clnt], idxs_test = None, dataset_name = args.dataset, FFC_net = FFC_net,need_ffc=need_ffc) #测试集没有放在client

            if idx == 0 :
                clnt_endtime = data_times[0]
                # Testing-------------------------------------------------
                [loss_tst, acc_tst] = local.evaluate(net = copy.deepcopy(FL_model).to(device), 
                                ell = args.local_ep , dataset_tst = test_data, 
                                dataset_tst_label = test_labels , net_server = net_server)                   
                print("'Federated Iteration %3d', 'loss_tst: %.4f', 'acc_tst: %.4f' "%(iter, loss_tst, acc_tst))      
                FL_acc.append(acc_tst)
                FL_loss.append(loss_tst)
                saved_itr +=1
                clnt_endtime = data_times[idx]
                writer.add_scalars('Accuracy/Test',
                {
                    'All Clients': FL_acc[saved_itr]
                }, clnt_endtime
                )
                writer.add_scalars('Loss/Test',
                {
                    'All Clients': FL_loss[saved_itr]
                }, clnt_endtime
                )           

                # Training test-------------------------------------------
                [loss_trn, acc_trn] = local.evaluate(net = copy.deepcopy(FL_model).to(device), 
                                ell = args.local_ep , dataset_tst = cent_x, 
                                dataset_tst_label = cent_y , net_server = net_server)                   
                print("'loss_train: %.4f', 'acc_trn: %.4f'" %( loss_trn, acc_trn))      
                FL_acc_trn.append(acc_trn)
                FL_loss_trn.append(loss_trn)       

                writer.add_scalars('Accuracy/Train',
                {
                    'All Clients': FL_acc_trn[saved_itr]
                }, clnt_endtime
                )
                writer.add_scalars('Loss/Train',
                {
                    'All Clients': FL_loss_trn[saved_itr]
                }, clnt_endtime
                ) 
           

            # Training ------------------
            [w_client, AN_params,smashed_list, AN_loss_train, AN_acc_train] = local.train(net = copy.deepcopy(clnt_models[clnt].to(device)))
            w_locals_client.append(copy.deepcopy(w_client))
            # AN_locals_client.append(copy.deepcopy(AN_params))

            AN_loss_train_list[clnt].append(AN_loss_train)
            AN_acc_train_list[clnt].append(AN_acc_train)

            # Freeze model
            for params in clnt_models[clnt].parameters():
                params.requires_grad = False

            for params in AN_net[clnt].parameters():
                params.requires_grad = False

            clnt_models[clnt].eval()
            AN_net[clnt].eval()

            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(w_client)))
            AN_net[clnt].load_state_dict(copy.deepcopy(dict(AN_params)))           
            clnt_smashed[clnt] = copy.deepcopy(smashed_list)
         
                                 
            for smashed_labels in smashed_list:#长度为64*20 即local_bs*local_ep
                smashed_data = smashed_labels[0]
                local_labels = smashed_labels[1]
                [global_gradient, net_server_update] = train_server(net_server, smashed_data, local_labels, device=device, lr=args.local_lr)
                clnt_glob_graditent[clnt] = global_gradient #最后一次的梯度
                net_server = copy.deepcopy(net_server_update) #有更新，但是很慢    

        
            if len(w_locals_client) == args.K:#iter%2==0,5,10
                args.async_lr = 1 if args.K % args.num_users == 0 else 0.1
                w_old = copy.deepcopy(FL_model.to(device).state_dict())
                # w_glob_client = FedBuff(w_old, w_locals_client, args.async_lr)  
                if args.Group == 2:
                    # FL_G4_params, FL_G2_params = FedBuff_layer(w_old,w_locals_client, args.async_lr)
                    FL_G4_params, FL_G2_params = FedBuff_del_inputlayer(w_old,w_locals_client, args.async_lr)
                    
                elif args.Group == 1:
                    FL_G4_params = FedBuff(w_old, w_locals_client, args.async_lr) 
                w_glob_client = copy.deepcopy(FL_G4_params)
                w_locals_client = []

                # Update client-side global model 
                FL_model.load_state_dict(w_glob_client) #只对客户端模型做FL
                if args.Group == 2:
                    if FL_G2_params !=0:
                        FL_G2.load_state_dict(FL_G2_params)  
                    else: FL_G2 = FL_G2


                iter += 1
                # Testing-------------------------------------------------
                [loss_tst, acc_tst] = local.evaluate(net = copy.deepcopy(FL_model).to(device), 
                                ell = args.local_ep , dataset_tst = test_data, 
                                dataset_tst_label = test_labels , net_server = net_server)                   
                print("'Federated Iteration %3d', 'loss_tst: %.4f', 'acc_tst: %.4f' "%(iter, loss_tst, acc_tst))      
                FL_acc.append(acc_tst)
                FL_loss.append(loss_tst)
                saved_itr +=1
                clnt_endtime = data_times[idx]
                writer.add_scalars('Accuracy/Test',
                {
                    'All Clients': FL_acc[saved_itr]
                }, clnt_endtime
                )
                writer.add_scalars('Loss/Test',
                {
                    'All Clients': FL_loss[saved_itr]
                }, clnt_endtime
                )           

                # Training test-------------------------------------------
                [loss_trn, acc_trn] = local.evaluate(net = copy.deepcopy(FL_model).to(device), 
                                ell = args.local_ep , dataset_tst = cent_x, 
                                dataset_tst_label = cent_y , net_server = net_server)                   
                print("'loss_train: %.4f', 'acc_trn: %.4f'" %( loss_trn, acc_trn))      
                FL_acc_trn.append(acc_trn)
                FL_loss_trn.append(loss_trn)       

                writer.add_scalars('Accuracy/Train',
                {
                    'All Clients': FL_acc_trn[saved_itr]
                }, clnt_endtime
                )
                writer.add_scalars('Loss/Train',
                {
                    'All Clients': FL_loss_trn[saved_itr]
                }, clnt_endtime
                ) 

                r_list = []
                s_list = copy.deepcopy(user_id)

            else:
                FL_model = FL_model #
                FL_G2 = FL_G2
                # FL_AN = FL_AN

            # if args.Group == 2:
            #     if clnt in idx_G4:
            #         clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_model.named_parameters())))
            #         # AN_net[clnt].load_state_dict(copy.deepcopy(dict(FL_AN.named_parameters())))
            #     elif clnt in idx_G2:
            #         clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_G2.named_parameters())))
            #         # AN_net[clnt].load_state_dict(copy.deepcopy(dict(FL_AN.named_parameters())))             
            # elif args.Group == 1:
            #         clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_model.named_parameters())))  

    if rule_iid == 'iid':
        file_tmp = 'cifar/40_iid_cifar_async_K/'
        if (not os.path.exists('save/%s' %(file_tmp))):
            os.makedirs('save/%s' %(file_tmp))            
    elif rule_iid == 'Noniid':
        file_tmp = 'cifar/40_Noniid_cifar_async_K/'
        if (not os.path.exists('save/%s' %(file_tmp))):
            os.makedirs('save/%s' %(file_tmp))
    file_name = 'save/' + file_tmp + 'SFL_acc_' + str(args.K) + '_' + str(args.async_lr) +  '_' + 'V2'
    write_info_to_txt(FL_acc, file_name)
    file_name = 'save/' + file_tmp + 'SFL_loss_' + str(args.K) + '_' + str(args.async_lr) +  '_' + 'V2'
    write_info_to_txt(FL_loss, file_name)
    file_name = 'save/' + file_tmp + 'SFL_acc_trn_' + str(args.K) + '_' + str(args.async_lr) +  '_' + 'V2' 
    write_info_to_txt(FL_acc_trn, file_name)
    file_name = 'save/' + file_tmp + 'SFL_loss_trn_' + str(args.K) + '_' + str(args.async_lr) +  '_' + 'V2'
    write_info_to_txt(FL_loss_trn, file_name)
    file_name = 'save/' + file_tmp + 'Client_train_acc_' + str(args.K) + '_' + str(args.async_lr) +  '_' + 'V2'
    write_info_to_txt(train_acc_all, file_name)
    file_name = 'save/' + file_tmp + 'Client_train_loss_' + str(args.K) + '_' + str(args.async_lr) +  '_' + 'V2' 
    write_info_to_txt(train_loss_all, file_name)    

    #==========================================   
    print("Training and Evaluation completed!")    


if __name__  == '__main__':
    for Group_type in [2,1]:
        for rule_iid in ['iid', 'Noniid']:
            for K in [40]:
                SFL_over_SA(rule_iid, K, Group_type)
    print('wait for check!')
