import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 4, 5, 6, 7'
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
from utils_SL.clients_side_SL import *
from server_side import *
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
    args.RUN = 'SL'
    # args.local_bs = 32
    # args.local_ep = 64
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
    program = "SL VGGNet"
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
    suffix += '_SL_' + str(args.num_users) + '_' + str(args.K) + '_' + str(args.dataset)
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
    FL_diff_1 = []
    FL_diff_2 = []
    net_server = []

    clnt_models = list(range(args.num_users))

    if args.Group == 1: # 同分割
        init_net_client_same = clnt_model_same()
        # pretrain
        load_pretrained(init_net_client_same, {x:x for x in [0, 2]}, 'features')

        FL_model = clnt_model_same()
        FL_model.load_state_dict(copy.deepcopy(dict(init_net_client_same.named_parameters())))

        net_server = server_model_same()
        load_pretrained(net_server, {x:x-5 for x in [5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]}, 'features')
        load_pretrained(net_server, {x:x for x in [0, 3]}, 'classifier')
        # pretrain
        
    elif args.Group == 2: # 不同分割
        init_net_client_diff_1 = clnt_model_diff_1()
        # pretrain
        load_pretrained(init_net_client_diff_1, {7:2}, 'features')
        init_net_client_diff_2 = clnt_model_diff_2()
        # pretrain
        load_pretrained(init_net_client_diff_2, {x:x for x in [0, 2, 5, 7]}, 'features')
        FL_diff_1 = clnt_model_diff_1()
        FL_diff_1.load_state_dict(copy.deepcopy(dict(init_net_client_diff_1.named_parameters())))

        FL_diff_2 = clnt_model_diff_2()
        FL_diff_2.load_state_dict(copy.deepcopy(dict(init_net_client_diff_2.named_parameters())))

        net_server = server_model_diff()
        # pretrain
        load_pretrained(net_server, {x:x-10 for x in [10, 12, 14, 17, 19, 21, 24, 26, 28]}, 'features')
        load_pretrained(net_server, {x:x for x in [0, 3]}, 'classifier')
    
    net_server.to(device)

    #---- Server side Arguments-----
    FL_loss = []
    FL_acc = []
    FL_loss_trn = []
    FL_acc_trn = []
    idx_diff_1 = []
    idx_diff_2 = []

    #%%---- 方案1：所有用户都分割一样的1层网络------
    if args.Group == 1:
        for clnt in user_id: #初始轮训分发
            #----load the global federated model------
            clnt_models[clnt] = clnt_model_same().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(init_net_client_same.named_parameters())))
    elif args.Group == 2:
    #---- 方案2：一半用户分割1层一半用户分割2层网络--
        m = max(int(args.frac * args.num_users), 1)
        idx_diff_1 = list((np.random.choice(user_id, m, replace = False)))
        idx_diff_2 = [id for id in user_id if (id not in idx_diff_1)]
        for clnt in idx_diff_1:
            clnt_models[clnt] = clnt_model_diff_1().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(init_net_client_diff_1.named_parameters())))           
        for clnt in idx_diff_2:
            clnt_models[clnt] = clnt_model_diff_2().to(device)
            clnt_models[clnt].load_state_dict(copy.deepcopy(dict(init_net_client_diff_2.named_parameters())))           

    #----内存释放----------
    del init_net_client_diff_1
    del init_net_client_diff_2
    del init_net_client_same

    #%%---从已经拿到初始网络之后开始训练程序—-----
    iter = 0        

    #%%Training test-------------------------------------------
    FL_test = []
    if args.Group == 1:
        FL_test = copy.deepcopy(FL_model).to(device)
    elif args.Group == 2:
        FL_test = copy.deepcopy(FL_diff_2).to(device)
    [loss_trn, acc_trn] = evaluate(net = FL_test, 
        dataset_tst = cent_x, 
        dataset_tst_label = cent_y , net_server = copy.deepcopy(net_server).to(device), device = device)                 
    print("'Split Iteration %3d','loss_train: %.4f', 'acc_trn: %.4f'" %( iter, loss_trn, acc_trn))         
    FL_acc_trn.append(acc_trn)
    FL_loss_trn.append(loss_trn)       

    writer.add_scalars('Accuracy/Train', { 'All Clients': FL_acc_trn[saved_itr]}, 0 )
    writer.add_scalars('Loss/Train', {'All Clients': FL_loss_trn[saved_itr]}, 0) 
    #%% Testing-------------------------------------------------
    [loss_tst, acc_tst] = evaluate(net = FL_test, 
        dataset_tst = data_obj.test_x, 
        dataset_tst_label = data_obj.test_y , net_server = copy.deepcopy(net_server).to(device), device = device)                     
    print("'Split Iteration %3d', 'loss_tst: %.4f', 'acc_tst: %.4f' "%(iter, loss_tst, acc_tst))      
    FL_acc.append(acc_tst)
    FL_loss.append(loss_tst)
    saved_itr +=1
    writer.add_scalars('Accuracy/Test',    {'All Clients': FL_acc[saved_itr]   }, 0)
    writer.add_scalars('Loss/Test', {'All Clients': FL_loss[saved_itr] }, 0 )  

    window_acc_trn_1 = 0
    window_acc_trn_2 = 0
    window_loss_trn_1 = 0
    window_loss_trn_2 = 0
    window_acc_tst_1 = 0
    window_acc_tst_2 = 0
    window_loss_tst_1 = 0
    window_loss_tst_2 = 0

    def calc_avg(curr, window):
        if len(window) >= 20:
            window = window[1:]
        window.append(curr)
        return 1.0 * sum(window) / len(window), window 

    #%%------按照星历表遍历----------------------------------------------------
    for idx, clnt in enumerate(user_list):    #按照星历表遍历       
        clnt_endtime = data_times[idx] #卫星离开的时间，星历表应该按照Endtime排序

        clnt_models[clnt].train()
        for params in clnt_models[clnt].parameters():
            params.requires_grad = True

        sats_clnt = Client(args, clnt_models[clnt], clnt, device, net_server,
                        dataset_train = clnt_x[clnt], dataset_test = None, idxs = clnt_y[clnt],
                        idxs_test = None, dataset_name = args.dataset) #测试集没有放在client
        [w_client, net_server_update] = sats_clnt.train(net = copy.deepcopy(clnt_models[clnt].to(device)))
        net_server = copy.deepcopy(net_server_update)

        if (idx + 1) % (args.num_users) == 0:  # 衰减步长 200
            args.local_lr = args.local_lr * (0.992 ** (4.0 * pow((idx + 1) / (args.num_users), 1.0 / 4.0)))
            args.local_lr = max(1e-3, args.local_lr)
        print(args.local_lr)

        iter+=1

        #%%Training test-------------------------------------------
        if args.Group == 1:
            FL_model.load_state_dict(copy.deepcopy(w_client))
            FL_test = copy.deepcopy(FL_model).to(device)
        elif args.Group == 2:
            if clnt in idx_diff_1:
                FL_diff_1.load_state_dict(copy.deepcopy(w_client))
                FL_test = copy.deepcopy(FL_diff_1).to(device)
            elif clnt in idx_diff_2:
                FL_diff_2.load_state_dict(copy.deepcopy(w_client))
                FL_test = copy.deepcopy(FL_diff_2).to(device)
        del w_client
        [loss_trn, acc_trn] = evaluate(net = FL_test, 
            dataset_tst = cent_x, 
            dataset_tst_label = cent_y , net_server = copy.deepcopy(net_server).to(device), device = device)
        # [loss_trn, window_loss_trn] = calc_avg(loss_trn, window_loss_trn)
        # [acc_trn, window_acc_trn] = calc_avg(acc_trn, window_acc_trn)  
        if clnt in idx_diff_1:
            window_loss_trn_1 = loss_trn
            window_acc_trn_1 = acc_trn
        elif clnt in idx_diff_2:
            window_loss_trn_2 = loss_trn
            window_acc_trn_2 = acc_trn
        loss_trn = (window_loss_trn_1 + window_loss_trn_2) / 2.0
        acc_trn = (window_acc_trn_1 + window_acc_trn_2) / 2.0               
        print("'Split Iteration %3d','loss_train: %.4f', 'acc_trn: %.4f'" % ( iter, loss_trn, acc_trn))      
        FL_acc_trn.append(acc_trn)
        FL_loss_trn.append(loss_trn)       

        writer.add_scalars('Accuracy/Train', { 'All Clients': FL_acc_trn[saved_itr]}, clnt_endtime )
        writer.add_scalars('Loss/Train', {'All Clients': FL_loss_trn[saved_itr]}, clnt_endtime) 
        #%% Testing-------------------------------------------------
        [loss_tst, acc_tst] = evaluate(net = FL_test, 
            dataset_tst = data_obj.test_x, 
            dataset_tst_label = data_obj.test_y , net_server = copy.deepcopy(net_server).to(device), device = device)
        if clnt in idx_diff_1:
            window_loss_tst_1 = loss_tst
            window_acc_tst_1 = acc_tst
        elif clnt in idx_diff_2:
            window_loss_tst_2 = loss_tst
            window_acc_tst_2 = acc_tst
        loss_tst = (window_loss_tst_1 + window_loss_tst_2) / 2.0
        acc_tst = (window_acc_tst_1 + window_acc_tst_2) / 2.0   
        # [loss_tst, window_loss_tst] = calc_avg(loss_tst, window_loss_tst)
        # [acc_tst, window_acc_tst] = calc_avg(acc_tst, window_acc_tst)                    
        print("'Split Iteration %3d', 'loss_tst: %.4f', 'acc_tst: %.4f' "%(iter, loss_tst, acc_tst))      
        FL_acc.append(acc_tst)
        FL_loss.append(loss_tst)
        saved_itr +=1
        writer.add_scalars('Accuracy/Test',    {'All Clients': FL_acc[saved_itr]   },clnt_endtime)
        writer.add_scalars('Loss/Test', {'All Clients': FL_loss[saved_itr] }, clnt_endtime)
        writer.add_scalars('LR', {'All Clients': args.local_lr}, clnt_endtime)

        if idx % 200 == 0:
            record(FL_acc, FL_loss, FL_acc_trn, FL_loss_trn, args, rule_iid)
        torch.cuda.empty_cache()

    record(FL_acc, FL_loss, FL_acc_trn, FL_loss_trn, args, rule_iid)
                # ==========================================
    print("Training and Evaluation completed!")

#%%--------Main-------------------------------------
#===================================================s

if __name__  == '__main__':
    for Group_type in [2]:
        for rule_iid in ['iid', 'Noniid']:
            for K in [1]:
                SFL_over_SA(rule_iid, K, Group_type)
    print('wait for check!')
