import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5,6,7'
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
from utils_centre.clients_side_centre import *
from utils_centre.server_side_centre import *
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
    args.RUN = 'centre'
    args.local_lr = 1e-3

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
    program = "Centre VGGNet"
    print(f"---------{program}----------")              
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #----- data asign -------------------
    rule_iid = rule_iid   
    data_path = 'Folder/'
    # data_obj = DatasetObject(dataset='mnist', n_client = args.num_users , seed=23, rule='iid', rule_arg = 0.6, unbalanced_sgm=0, data_path=data_path)
    data_obj = DatasetObject(dataset=args.dataset, n_client = 1 , seed=23, rule= rule_iid, rule_arg = 0.6, unbalanced_sgm=0, data_path=data_path)
    
    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y
    cent_x = np.concatenate(clnt_x, axis = 0)#训练集
    cent_y = np.concatenate(clnt_y, axis = 0)

    #%%---- Tensorboard for checking results----------------------------------------
    suffix = 'Centre_' + str(args.Group) 
    suffix += '_centre_' + str(1) + '_' + str(args.K) + '_' + str(args.dataset)
    suffix += '_LR%f_BS%d_E%d_' %(args.local_lr, args.local_bs, args.local_ep)
    data_path_tb = 'Folder/'
    if (not os.path.exists('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))):
        # os.mkdir('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
        os.makedirs('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
   
    saved_itr = -1 #用作结果记录
    # writer = SummaryWriter('%sRuns/%s/%s' %(data_path_tb, data_obj.instance_name, suffix))
   
    #%%----- Split model Client side ----
    torch.manual_seed(37)

    FL_model = []

    # clnt_models = list(range(args.num_users))
    net_server = models.vgg16(pretrained=True)
    in_features = net_server.classifier[6].in_features
    net_server.classifier[6] = nn.Linear(in_features, 10)

    FL_model = models.vgg16(pretrained=True)
    in_features = FL_model.classifier[6].in_features
    FL_model.classifier[6] = nn.Linear(in_features, 10)

    #---- Server side Arguments-----
    FL_loss = []
    FL_acc = []
    FL_loss_trn = []
    FL_acc_trn = []

    #%%---- 方案1：所有用户都分割一样的1层网络------
    #%%---从已经拿到初始网络之后开始训练程序—-----
    iter = 0        

    #%%Training test-------------------------------------------
    FL_test = []
    if args.Group == 1:
        FL_test = copy.deepcopy(FL_model).to(device)
    [loss_trn, acc_trn] = evaluate(net = FL_test, 
        dataset_tst = cent_x, 
        dataset_tst_label = cent_y, device = device)                 
    print("'Federated Iteration %3d','loss_train: %.4f', 'acc_trn: %.4f'" %(iter, loss_trn, acc_trn))         
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

    server_iter = 0

    #%%------按照星历表遍历----------------------------------------------------
    for idx, clnt in enumerate(user_list):    #按照星历表遍历       
        clnt_endtime = data_times[idx] #卫星离开的时间，星历表应该按照Endtime排序

        net_server.train()
        for params in net_server.parameters():
            params.requires_grad = True

        sats_clnt = Client(args, net_server, clnt, device,
                        dataset_train = clnt_x[0], dataset_test = None, idxs = clnt_y[0],
                        idxs_test = None, dataset_name = args.dataset) #测试集没有放在client
        w_client = sats_clnt.train(net = copy.deepcopy(net_server.to(device)))     

        server_iter = server_iter + 1

        FL_model.load_state_dict(w_client)


        if (idx + 1) % (args.num_users/2) == 0:  # 衰减步长 200
            args.local_lr = args.local_lr * (0.992 ** (4.0 * pow((idx + 1) / (args.num_users/2), 1.0 / 4.0)))
            args.local_lr = max(1e-4, args.local_lr)
        print(args.local_lr)
        iter += 1

        #%%Training test-------------------------------------------
        if args.Group == 1:
            FL_test = copy.deepcopy(FL_model).to(device)
        [loss_trn, acc_trn] = evaluate(net = FL_test, 
            dataset_tst = cent_x, 
            dataset_tst_label = cent_y, device = device)                 
        print("'Federated Iteration %3d','loss_train: %.4f', 'acc_trn: %.4f'" % ( iter, loss_trn, acc_trn))      
        FL_acc_trn.append(acc_trn)
        FL_loss_trn.append(loss_trn)       

        # writer.add_scalars('Accuracy/Train', { 'All Clients': FL_acc_trn[saved_itr]}, clnt_endtime )
        # writer.add_scalars('Loss/Train', {'All Clients': FL_loss_trn[saved_itr]}, clnt_endtime) 
        #%% Testing-------------------------------------------------
        [loss_tst, acc_tst] = evaluate(net = FL_test, 
            dataset_tst = data_obj.test_x, 
            dataset_tst_label = data_obj.test_y, device = device)                     
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

            net_server.load_state_dict(copy.deepcopy(dict(FL_model.named_parameters())))

        if idx % 200 == 0:
            record(FL_acc, FL_loss, FL_acc_trn, FL_loss_trn, args, rule_iid)

    record(FL_acc, FL_loss, FL_acc_trn, FL_loss_trn, args, rule_iid)
                # ==========================================
    print("Training and Evaluation completed!")

#%%--------Main-------------------------------------
#===================================================s

if __name__  == '__main__':
    for Group_type in [1]:
        for rule_iid in ['iid', 'Noniid']:
            for K in [1]:
                SFL_over_SA(rule_iid, K, Group_type)
    print('wait for check!')
