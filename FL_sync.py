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
from torchvision import models

from utils_dataset import *
from utils_model import *
from utils_general import *
from utils_args import args_parser
from utils_FL.clients_side_FL import *
from utils_FL.server_side_FL import *
import math

  
from tensorboardX import SummaryWriter

#---------------------------------------------
#%%----- main process ----
def SFL_over_SA(rule_iid ,K, Group):
    #------Random seed-----
    random_seed()
    args = args_parser()
    args.K = K
    args.Group = Group
    args.RUN = 'FL_sync'
    args.local_bs = 16
    args.local_ep = 128

    #%%-----------------------------------
    # The Core Process
    #-------------------------------------
    """
    1.read SA-GS connect data from csv total five days
    # """
    data_csv = pd.read_csv('./20SA_15DAY_endtime.csv')#按照EndTime排序
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
    program = "FL VGGNet"
    print(f"---------{program}----------")              
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    
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
    suffix = 'Sync_' + str(args.Group) 
    suffix += '_FL_' + str(args.num_users) + '_' + str(args.K) + '_' + str(args.dataset)
    suffix += '_LR%f_BS%d_E%d_' %(args.local_lr, args.local_bs, args.local_ep)
    data_path_tb = 'Folder/'
    if (not os.path.exists('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))):
        # os.mkdir('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
        os.makedirs('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
   
    saved_itr = -1 #用作结果记录
    # writer = SummaryWriter('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
   
    #%%----- Split model Client side ----
    # if args.dataset == 'mnist':
    #     clnt_model_func = lambda: CNNmnist_client_side()
    if args.dataset =='Cifar10' or 'CIFAR10':
        clnt_model_same = lambda: vgg16()

    torch.manual_seed(37)

    init_net_client_same = []
    FL_model = []

    clnt_models = list(range(args.num_users))

    if args.Group == 1: # 同分割
        init_net_client_same = clnt_model_same()
        # pretrain
        load_pretrained(init_net_client_same, {x:x for x in [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]}, 'features')
        load_pretrained(init_net_client_same, {x:x for x in [0, 3]}, 'classifier')

        FL_model = clnt_model_same()
        FL_model.load_state_dict(copy.deepcopy(dict(init_net_client_same.named_parameters())))

    #---- Server side Arguments-----
    FL_loss = []
    FL_acc = []
    FL_loss_trn = []
    FL_acc_trn = []
    w_locals_client = []

    #%%---- 方案1：所有用户都分割一样的1层网络------
    if args.Group == 1:
        for clnt in user_id: #初始轮训分发
            #----load the global federated model------
            clnt_models[clnt] =  clnt_model_same()
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(init_net_client_same.named_parameters())))
    #----内存释放----------
    del init_net_client_same

    #%%---从已经拿到初始网络之后开始训练程序—-----
    iter = 0      

    loss_trn = 0
    acc_trn = 0
    loss_tst = 0
    acc_tst = 0  

    #%%Training test-------------------------------------------
    FL_test = []
    if args.Group == 1:
        FL_test = copy.deepcopy(FL_model).to(device)
    [loss_trn, acc_trn] = evaluate(net = FL_test, 
        dataset_tst = cent_x, 
        dataset_tst_label = cent_y, device = device)                 
    print("'Federated Iteration %3d','loss_train: %.4f', 'acc_trn: %.4f'" %( iter, loss_trn, acc_trn))         
    FL_acc_trn.append(acc_trn)
    FL_loss_trn.append(loss_trn)       

    # writer.add_scalars('Accuracy/Train', { 'All Clients': FL_acc_trn[saved_itr]}, 0 )
    # writer.add_scalars('Loss/Train', {'All Clients': FL_loss_trn[saved_itr]}, 0) 
    #%% Testing-------------------------------------------------
    [loss_tst, acc_tst] = evaluate(net = FL_test, 
        dataset_tst = data_obj.test_x, 
        dataset_tst_label = data_obj.test_y, device = device)                     
    print("'Federated Iteration %3d', 'loss_tst: %.4f', 'acc_tst: %.4f' "%(iter, loss_tst, acc_tst))      
    FL_acc.append(acc_tst)
    FL_loss.append(loss_tst)
    saved_itr +=1
    # writer.add_scalars('Accuracy/Test',    {'All Clients': FL_acc[saved_itr]   }, 0)
    # writer.add_scalars('Loss/Test', {'All Clients': FL_loss[saved_itr] }, 0 )  

    s_list = [] #记录分发
    r_list = copy.deepcopy(user_id)   #记录待接收
    #如果没有新的更新就空转 ：防止不同的客户端训练频次不同，容易过拟合
    s_num = 0
    server_iter = 0  # 学习率衰减控制

    #%%------按照星历表遍历----------------------------------------------------
    for idx, clnt in enumerate(user_list):    #按照星历表遍历       
        clnt_endtime = data_times[idx] #卫星离开的时间，星历表应该按照Endtime排序

        if clnt in s_list: #分发，注意到第一轮的分发可是统一初始化
            if args.Group == 1:
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_model.named_parameters())))

            s_list.remove(clnt)
            r_list.append(clnt)#分发过了客户端可以在下次可见先回收
            s_num += 1
            if (s_num > 0) and (s_num % args.num_users == 0):
                s_list = []
                # r_list = copy.deepcopy(user_id)        

        elif clnt in r_list: #接收
            r_list.remove(clnt)
            print(f'Client_{clnt} remove from r_list, remain {r_list}')
            clnt_models[clnt].train()
            for params in clnt_models[clnt].parameters():
                params.requires_grad = True

            # Train Client
            sats_clnt = Client(args, clnt, device,
                            dataset_train = clnt_x[clnt], dataset_test = None,
                            idxs = clnt_y[clnt], idxs_test = None,
                            dataset_name = args.dataset) #测试集没有放在client
            w_client = sats_clnt.train(net = copy.deepcopy(clnt_models[clnt].to(device)))
            w_locals_client.append(copy.deepcopy(w_client))
            del sats_clnt, w_client
            if (server_iter + 1) % 10 == 0:  # 衰减步长 200
                args.local_lr = args.local_lr * (0.992 ** (4.0 * pow((server_iter + 1.0) / 10.0, 1.0 / 4.0)))
                args.local_lr = max(1e-2, args.local_lr)
            print(args.local_lr)
            server_iter = server_iter + 1

            # 测试 FL_model + server_model            
            # 这里也可以写成 if r_list == 0
            if len(w_locals_client) >= args.K and (args.K == args.num_users):             #iter%2==0,5,10

                print('观测聚合内容长度', len(w_locals_client))#一直是20个，不重复的user提供的
                args.async_lr = 1 if args.K % args.num_users == 0 else 0.1
                # TODO
                if args.Group == 1:
                    w_old = copy.deepcopy(FL_model.to(device).state_dict()) 
                    FL_params = FedBuff(w_old, w_locals_client, args.async_lr)
                    FL_model.load_state_dict(FL_params) #只对客户端模型做FL
                    del w_old, FL_params

                #清除Buff-----------------------------
                w_locals_client = []
                #更新r_list和s_list-------------------
                r_list = []
                s_list = copy.deepcopy(user_id)                    

                iter += 1 #检验FL次数  

                if args.Group == 1:
                    FL_test = copy.deepcopy(FL_model)
                    FL_test = nn.DataParallel(FL_test)
                    FL_test.to(device)
                [loss_trn, acc_trn] = evaluate(net = FL_test, 
                    dataset_tst = cent_x, 
                    dataset_tst_label = cent_y, device = device)    
                [loss_tst, acc_tst] = evaluate(net = FL_test, 
                    dataset_tst = data_obj.test_x, 
                    dataset_tst_label = data_obj.test_y, device = device)  
                FL_test = []
                
            else:
                FL_model = FL_model

        # iter += 1
        #%%Training test-------------------------------------------  
        print("'Federated Iteration %3d','loss_train: %.4f', 'acc_trn: %.4f'" % ( iter, loss_trn, acc_trn))      
        FL_acc_trn.append(acc_trn)
        FL_loss_trn.append(loss_trn)       

        # writer.add_scalars('Accuracy/Train', { 'All Clients': FL_acc_trn[saved_itr]}, clnt_endtime )
        # writer.add_scalars('Loss/Train', {'All Clients': FL_loss_trn[saved_itr]}, clnt_endtime) 
        #%% Testing-------------------------------------------------                 
        print("'Federated Iteration %3d', 'loss_tst: %.4f', 'acc_tst: %.4f' "%(iter, loss_tst, acc_tst))      
        FL_acc.append(acc_tst)
        FL_loss.append(loss_tst)
        saved_itr +=1
        # writer.add_scalars('Accuracy/Test',    {'All Clients': FL_acc[saved_itr]   },clnt_endtime)
        # writer.add_scalars('Loss/Test', {'All Clients': FL_loss[saved_itr] }, clnt_endtime)
        # writer.add_scalars('LR', {'All Clients': args.local_lr}, clnt_endtime)

        #%%---------------------------------回传-------------------------------------------------
        if args.Group == 1:
            # Freeze model
            for params in FL_model.parameters():
                params.requires_grad = False
            
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(FL_model.named_parameters())))

        if idx % 200 == 0:
            record(FL_acc, FL_loss, FL_acc_trn, FL_loss_trn, args, rule_iid)
        torch.cuda.empty_cache()
    #%----------------------记录---------------------------------------------------------------
    record(FL_acc, FL_loss, FL_acc_trn, FL_loss_trn, args, rule_iid)

    #==========================================   
    print("Training and Evaluation completed!")

#%%--------Main-------------------------------------
#===================================================s

if __name__  == '__main__':
    for Group_type in [1]:
        for rule_iid in ['iid', 'Noniid']:
            for K in [20]:
                SFL_over_SA(rule_iid, K, Group_type)
    print('wait for check!')
