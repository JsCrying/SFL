import torch
import copy
import math
from collections import OrderedDict
from torch import nn
from utils_general import *
from torch.utils.data import DataLoader
from utils_dataset import Dataset
# import torch.optim.lr_scheduler as lr_scheduler
import sys

#%%-----Server side----
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))#除法
    return w_avg

def FedAsyncPoly(w_old, w_delta, recv_list, server_iter, args):
    w_buff = copy.deepcopy(w_old)
    for i in range(len(recv_list)):
        print(f'server_iter-recv_list[i]={server_iter-recv_list[i]}')
        lr = args.async_lr * pow(1 + server_iter - recv_list[i], args.poly_deg)
        for k in w_delta[0].keys():
            w_buff[k] = torch.mul(w_buff[k], 1-lr) + torch.mul(w_delta[i][k], lr)
    return w_buff

def FedAsyncPoly_diff(w_old_1, w_old_2, w_delta, recv_list, server_iter, args):
    # 不同分割模型对应的参数量
    params_1 = 4
    params_2 = 8
    # w_delta 里不同分割模型归类的索引
    idx_1 = [i for i in range(len(w_delta)) if len(w_delta[i]) == params_1]
    idx_2 = [i for i in range(len(w_delta)) if len(w_delta[i]) == params_2]

    # 切割模型分为 各自层 + 公共层（两种模型里都是最后一层）
    w_delta_1 = [
        {item:w_delta[i][item] for item in (list(OrderedDict(w_delta[i]))[:-2])} for i in idx_1
    ]
    w_delta_2 = [
        {item:w_delta[i][item] for item in (list(OrderedDict(w_delta[i]))[:-2])} for i in idx_2
    ]
    w_delta_common = []

    # 聚合各自层
    w_buff_1 = copy.deepcopy(w_old_1)
    w_buff_2 = copy.deepcopy(w_old_2)
    recv_list_1 = [recv_list[i] for i in idx_1]
    recv_list_2 = [recv_list[i] for i in idx_2]
    if len(w_delta_1):
        w_buff_1 = FedAsyncPoly(w_old_1, w_delta_1, recv_list_1, server_iter, args)
    if len(w_delta_2):
        w_buff_2 = FedAsyncPoly(w_old_2, w_delta_2, recv_list_2, server_iter, args)

    # 聚合公共层
    # 参数名称统一
    if len(idx_1):
        param = list(OrderedDict(w_delta[idx_1[0]]))
        w_delta_common = [ {
                param[j]: w_delta[i][list(OrderedDict(w_delta[i]))[j]]
                for j in [-2, -1]
            } 
            for i in range(len(w_delta))
        ]
        w_buff_common = FedAsyncPoly(w_old_1, w_delta_common, recv_list, server_iter, args) 
    elif len(idx_2):
        param = list(OrderedDict(w_delta[idx_2[0]]))
        w_delta_common = [ {
                param[j]: w_delta[i][list(OrderedDict(w_delta[i]))[j]]
                for j in [-2, -1]
            } 
            for i in range(len(w_delta))
        ]
        w_buff_common = FedAsyncPoly(w_old_2, w_delta_common, recv_list, server_iter, args) 

    # 根据聚合结果调整公共层旧模型参数
    for i in [-2, -1]:
        new_common = w_buff_common[list(OrderedDict(w_buff_common))[i]]
        w_buff_1[list(OrderedDict(w_buff_1))[i]] = new_common
        w_buff_2[list(OrderedDict(w_buff_2))[i]] = new_common

    
    return w_buff_1, w_buff_2


def FedBuff(w_old,w_delta,lr):
    w_buff = copy.deepcopy(w_old)
    w_avg_delta = copy.deepcopy(w_delta[0])
    for k in w_avg_delta.keys():
        for i in range(1,len(w_delta)):
            w_avg_delta[k] += w_delta[i][k]
        w_avg_delta[k] = torch.div(w_avg_delta[k],len(w_delta))
        w_buff[k] = torch.mul(w_buff[k],(1-lr)) + torch.mul(w_avg_delta[k],lr)
    return w_buff            

def FedBuff_diff(w_old_1, w_old_2, w_delta, lr):
    # 不同分割模型对应的参数量
    params_1 = 4
    params_2 = 8
    # w_delta 里不同分割模型归类的索引
    idx_1 = [i for i in range(len(w_delta)) if len(w_delta[i]) == params_1]
    idx_2 = [i for i in range(len(w_delta)) if len(w_delta[i]) == params_2]

    # 切割模型分为 各自层 + 公共层（两种模型里都是最后一层）
    w_delta_1 = [
        {item:w_delta[i][item] for item in (list(OrderedDict(w_delta[i]))[:-2])} for i in idx_1
    ]
    w_delta_2 = [
        {item:w_delta[i][item] for item in (list(OrderedDict(w_delta[i]))[:-2])} for i in idx_2
    ]
    w_delta_common = []

    # 聚合各自层
    w_buff_1 = copy.deepcopy(w_old_1)
    w_buff_2 = copy.deepcopy(w_old_2)
    if len(w_delta_1):
        w_buff_1 = FedBuff(w_old_1, w_delta_1, lr)
    if len(w_delta_2):
        w_buff_2 = FedBuff(w_old_2, w_delta_2, lr)

    # 聚合公共层
    # 参数名称统一
    if len(idx_1):
        param = list(OrderedDict(w_delta[idx_1[0]]))
        w_delta_common = [ {
                param[j]: w_delta[i][list(OrderedDict(w_delta[i]))[j]]
                for j in [-2, -1]
            } 
            for i in range(len(w_delta))
        ]
        w_buff_common = FedBuff(w_old_1, w_delta_common, lr) 
    elif len(idx_2):
        param = list(OrderedDict(w_delta[idx_2[0]]))
        w_delta_common = [ {
                param[j]: w_delta[i][list(OrderedDict(w_delta[i]))[j]]
                for j in [-2, -1]
            } 
            for i in range(len(w_delta))
        ]
        w_buff_common = FedBuff(w_old_2, w_delta_common, lr) 

    # 根据聚合结果调整公共层旧模型参数
    for i in [-2, -1]:
        new_common = w_buff_common[list(OrderedDict(w_buff_common))[i]]
        w_buff_1[list(OrderedDict(w_buff_1))[i]] = new_common
        w_buff_2[list(OrderedDict(w_buff_2))[i]] = new_common

    
    return w_buff_1, w_buff_2

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc


def calculate_accuracy_CPU(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    #转到CPU
    preds = preds.cpu().numpy()
    bs_y = y.cpu().numpy().astype(np.int32)
    # correct = preds.eq(y.view_as(preds)).sum()
    correct = np.sum(preds == bs_y)
    acc_bs = correct
    # acc = 100.00 *correct.float()/preds.shape[0]
    return acc_bs

def evaluate_server_V2(fx_client, y, device):

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        #--- loss and acc ---
        loss = loss_fn(fx_client+1e-8,y.reshape(-1).long())
        # loss = loss/list(y.size())[0]
        acc_test = calculate_accuracy_CPU(fx_client, y)
    return loss, acc_test

def evaluate(net, dataset_tst, dataset_tst_label, device):

    acc_overall = 0; loss_overall = 0
    batch_size = min(1000, dataset_tst.shape[0])#选择batch_size小于2000
    ldr_test = DataLoader(Dataset(dataset_tst, dataset_tst_label,train=False, dataset_name = 'CIFAR10'),
                    batch_size=batch_size,shuffle=False)
    #TODO: dataset_name名字要根据具体的修改
    #TODO:test bs可以传输args.bs进来
    net.eval()
    net = net.to(device)
    tst_gene_iter = ldr_test.__iter__()
    n_tst = dataset_tst.shape[0]
    with torch.no_grad():           
        for count_tst in range(int(np.ceil((n_tst)/batch_size))):
            images,labels = tst_gene_iter.__next__()
            images = images.to(device)
            labels = labels.to(device)
            fx = net(images)
            # Sending activations to server 
            loss_tmp, acc_tmp = evaluate_server_V2(fx, labels, device)

            loss_overall += loss_tmp.item()
            acc_overall += acc_tmp.item()

    loss_overall /= n_tst
    w_decay = 1e-3 #TODO:w_decay修改
    # if w_decay != None:
    #     # Add L2 loss
    #     params = get_mdl_params([net], n_par=None)
    #     loss_overall += w_decay/2 * np.sum(params * params)
        
    net.train()
    # acc_overall = acc_overall / (count_tst+1)
    acc_overall = 100.00*acc_overall / n_tst
    return loss_overall, acc_overall              

def get_mdl_params(model_list, n_par=None):
    
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0 #参数个数？199210
        for name, param in exp_mdl.named_parameters():#named_parameters() vs parameters()前者给出网络层的名字和参数的迭代器，后者给出参数迭代器
            n_par += len(param.data.reshape(-1))
    
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1) #为什么要把数据cpu()
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)