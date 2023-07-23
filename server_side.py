import torch
import copy
from torch import nn
from utils_general import *
from torch.utils.data import DataLoader
from utils_dataset import Dataset
# import torch.optim.lr_scheduler as lr_scheduler

#%%-----Server side----
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))#除法
    return w_avg

def FedBuff(w_old,w_delta,lr):
    w_buff = copy.deepcopy(w_old)
    w_avg_delta = copy.deepcopy(w_delta[0])
    for k in w_avg_delta.keys():
        for i in range(1,len(w_delta)):
            w_avg_delta[k] += w_delta[i][k]
        w_avg_delta[k] = torch.div(w_avg_delta[k],len(w_delta))
        w_buff[k] = torch.mul(w_buff[k],(1-lr)) + torch.mul(w_avg_delta[k],lr)
    #len=K=10,lr=1时候为同步
    return w_buff            


#-------方案1：切3+4层，加了一层卷积层-------
def FedBuff_layer(w_old,w_delta,lr):
    w_buff = copy.deepcopy(w_old)
    len_max = 8
    len_min = 4
    G2 = []
    G4 = []
    for i in range(len(w_delta)):
        len_tmp = len(w_delta[i])
        if len_tmp == len_max: G4.append(w_delta[i])
        elif len_tmp == len_min: G2.append(w_delta[i])

    K_G4 = len(G4)
    if K_G4 > 0:   
        w_avg_G4 = copy.deepcopy(G4[0])        
        keys_list = list(w_avg_G4.keys())
        for j,k in enumerate(keys_list):
            for i in range(1,K_G4):
                w_avg_G4[k] += G4[i][k]
            if j <= 3: 
                w_avg_G4[k] = torch.div(w_avg_G4[k],K_G4)
                w_buff[k] = torch.mul(w_buff[k],(1-lr)) + torch.mul( w_avg_G4[k],lr)

        K_G2 = len(G2)
        if K_G2 == 0 : 
            for k in keys_list[4:]: 
                w_avg_G4[k] = torch.div(w_avg_G4[k],K_G4)
                w_buff[k] = torch.mul(w_buff[k],(1-lr)) + torch.mul( w_avg_G4[k],lr)
        elif K_G2 > 0:
            keys_list2 = list(G2[0].keys())
            for j, k in enumerate(keys_list2):
                for i in range(K_G2):
                    w_avg_G4[keys_list[j+4]] += G2[i][k]
                w_avg_G4[keys_list[j+4]] = torch.div(w_avg_G4[keys_list[j+4]],K_G2+K_G4)
                w_buff[keys_list[j+4]] = torch.mul(w_buff[keys_list[j+4]],(1-lr)) + torch.mul(w_avg_G4[keys_list[j+4]],lr)

    elif K_G4 == 0:
        K_G2 = len(G2)
        w_avg_G2 = copy.deepcopy(G2[0])
        keys_list = list(w_old.keys())
        keys_list2 = list(w_avg_G2.keys())
        for j ,k in enumerate(keys_list2):
            for i in range(1,K_G2):
                w_avg_G2[k] += G2[i][k]
            w_avg_G2[k] = torch.div(w_avg_G2[k],K_G2)
            w_buff[keys_list[j+4]] = torch.mul(w_buff[keys_list[j+4]],(1-lr)) + torch.mul(w_avg_G2[k],lr)       

    FL_G4 = w_buff
    if K_G2 != 0:
        FL_G2 = copy.deepcopy(G2[0])
        for j,k in enumerate(keys_list2):
            FL_G2[k] = w_buff[keys_list[j+4]]
    else: FL_G2 = 0
    return FL_G4, FL_G2    

    
#---------方案2：切第一层输入层+第4层---------
def FedBuff_del_inputlayer(w_old,w_delta,lr):
    w_buff = copy.deepcopy(w_old)
    w_delta = copy.deepcopy(w_delta)

    len_max = 8
    len_min = 4
    G2 = []
    G4 = []
    for i in range(len(w_delta)):
        len_tmp = len(w_delta[i])
        if len_tmp == len_max: G4.append(w_delta[i])
        elif len_tmp == len_min: G2.append(w_delta[i])

    K_G2 = len(G2)
    K_G4 = len(G4)
    if K_G2 == 0:
        w_buff = FedBuff(w_buff,w_delta,lr)
        FL_G4 = w_buff
        FL_G2 = 0
        return FL_G4, FL_G2
    elif K_G4 == 0:
        w_avg_G2 = copy.deepcopy(G2[0])
        keys_list = list(w_old.keys())
        keys_list2 = list(w_avg_G2.keys())
        for j ,k in enumerate(keys_list2):
            for i in range(1,K_G2):
                w_avg_G2[k] += G2[i][k]
            w_avg_G2[k] = torch.div(w_avg_G2[k],K_G2)
            if j >= 2: #只聚合第4层的内容到全局
                w_buff[keys_list[j+4]] = torch.mul(w_buff[keys_list[j+4]],(1-lr)) + torch.mul(w_avg_G2[k],lr)       
                w_avg_G2[k] = w_buff[keys_list[j+4]]
        FL_G4 = w_buff
        FL_G2 = w_avg_G2
        return FL_G4, FL_G2
    else:
        w_avg_G2 = copy.deepcopy(G2[0])
        keys_list2 = list(w_avg_G2.keys())
        for j ,k in enumerate(keys_list2):
            for i in range(1,K_G2):
                w_avg_G2[k] += G2[i][k]
            if j <= 1:#Input 层
                w_avg_G2[k] = torch.div(w_avg_G2[k],K_G2)

        w_avg_G4 = copy.deepcopy(G4[0])        
        keys_list = list(w_avg_G4.keys())
        for j,k in enumerate(keys_list):
            for i in range(1,K_G4):
                w_avg_G4[k] += G4[i][k]
            if j < 6:
                w_avg_G4[k] = torch.div(w_avg_G4[k],K_G4)
                w_buff[k] = torch.mul(w_buff[k],(1-lr)) + torch.mul(w_avg_G4[k],lr)
            elif j>=6 :
                w_avg_G4[k] += w_avg_G2[keys_list[j-4]]
                w_avg_G4[k] = torch.div(w_avg_G4[k],K_G2+K_G4)
                w_buff[k] = torch.mul(w_buff[k],(1-lr)) + torch.mul(w_avg_G4[k],lr)
                w_avg_G2[keys_list[j-4]] = w_buff[k]
        FL_G4 = w_buff
        FL_G2 = w_avg_G2
        return FL_G4, FL_G2



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

#TODO： Server-side function associated with Training 给每一个用户都分配一个server
def train_server(net_server, fx_client, y, device, lr):
  
    net_server = net_server.to(device)
    net_server.train()
    # optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr) 
    optimizer_server = torch.optim.SGD(net_server.parameters(), lr = lr, weight_decay = 1e-3, momentum = 0.9)#0.9
    criterion = nn.CrossEntropyLoss(reduction='sum')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_server, step_size = 1, gamma = 0.9)

    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_server = net_server(fx_client)
    
    # calculate loss
    loss = criterion(fx_server+1e-8,y.reshape(-1).long())
    loss = loss/list(y.size())[0]

    assert torch.isnan(loss).sum() == 0, print(loss)
    # calculate accuracy
    # acc = calculate_accuracy(fx_server, y)
    # acc = calculate_accuracy_CPU(fx_server, y)
    
    #--------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    torch.nn.utils.clip_grad_norm_(parameters=net_server.parameters(), max_norm=3)#10,5,3
    optimizer_server.step()   
    # print("'Server-side LR: %.4f' "%(scheduler.get_lr()[0]))      
    #     
    scheduler.step()    

    net_server.eval()
    for params in net_server.parameters():
        params.required_grad = False

    return net_server, dfx_client


def evaluate_server_V2(fx_client, y, net_server, device):

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    net = copy.deepcopy(net_server).to(device)
    net.eval()
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        #--- forward prop ---
        fx_server = net(fx_client)
        #--- loss and acc ---
        loss = loss_fn(fx_server+1e-8,y.reshape(-1).long())
        # loss = loss/list(y.size())[0]
        acc_test = calculate_accuracy_CPU(fx_server,y)
    return loss, acc_test

def evaluate(net,dataset_tst, dataset_tst_label, net_server,device):

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
            loss_tmp, acc_tmp = evaluate_server_V2(fx, labels, net_server, device)

            loss_overall += loss_tmp.item()
            acc_overall += acc_tmp.item()

    loss_overall /= n_tst
    w_decay = 1e-3 #TODO:w_decay修改
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([net], n_par=None)
        loss_overall += w_decay/2 * np.sum(params * params)
        
    net.train()
    # acc_overall = acc_overall / (count_tst+1)
    acc_overall = 100.00*acc_overall / n_tst
    return loss_overall, acc_overall    
# def evaluate(self, net, ell, dataset_tst, dataset_tst_label, net_server):

#     acc_overall = 0; loss_overall = 0;
#     # loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
#     # batch_size = min(6000, dataset_tst.shape[0])
#     # batch_size = min(2000, dataset_tst.shape[0])#选择batch_size小于2000
#     batch_size = min(1000, dataset_tst.shape[0])#选择batch_size小于2000
#     self.ldr_test = DataLoader(Dataset(dataset_tst, dataset_tst_label,train=False, dataset_name = self.dataset_name),
#                     batch_size=batch_size,shuffle=False)

#     #TODO:test bs可以传输args.bs进来
#     net.eval()
#     net = net.to(self.device)
#     tst_gene_iter = self.ldr_test.__iter__()
#     n_tst = dataset_tst.shape[0]
#     with torch.no_grad():           
#         for count_tst in range(int(np.ceil((n_tst)/batch_size))):
#             images,labels = tst_gene_iter.__next__()
#             images = images.to(self.device)
#             labels = labels.to(self.device)
#             fx = net(images)
#             # Sending activations to server 
#             loss_tmp, acc_tmp = evaluate_server_V2(fx, labels, net_server, self.device)

#             loss_overall += loss_tmp.item()
#             acc_overall += acc_tmp.item()

#     loss_overall /= n_tst
#     w_decay = 1e-3 #TODO:w_decay修改
#     if w_decay != None:
#         # Add L2 loss
#         params = get_mdl_params([net], n_par=None)
#         loss_overall += w_decay/2 * np.sum(params * params)
        
#     net.train()
#     # acc_overall = acc_overall / (count_tst+1)
#     acc_overall = 100.00*acc_overall / n_tst
#     return loss_overall, acc_overall            


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