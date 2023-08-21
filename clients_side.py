from torch.utils.data import DataLoader
from utils_dataset import Dataset
import torch
import copy
from server_side import *
from torch.utils import data

import sys

#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # image, label = self.dataset[self.idxs[item]]
        label = self.idxs[item]
        image = self.dataset[label]
        #TODO:验证这里的迭代
        return image, label

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, args, clnt_model, idx, device, AN_net, recv_iter=0, do_srv2clnt_grad=1,
                 dataset_train = None, dataset_test = None, idxs = None, idxs_test = None, dataset_name = 'CIFAR10'):
        self.args = args
        self.clnt_model = clnt_model
        self.idx = idx
        self.lr = args.local_lr / args.diff_lr # 差异化与server LR
        self.local_ep = args.local_ep
        self.device = device
        self.AN_net = AN_net
        self.dataset_train = dataset_train
        self.idxs = idxs # tain labels
        
        self.dataset_test = dataset_test
        self.idxs_test = idxs_test
        self.dataset_name = dataset_name

        self.AN_loss_train = []#以batch存
        self.AN_acc_train = []
        self.bs = args.local_bs
        self.dataset_name = args.dataset

        self.do_srv2clnt_grad = do_srv2clnt_grad
        self.fxs = []
        self.AN_grads = []

        self.recv_iter = recv_iter # 收到来自server模型的服务器周期

    def do_grad(self):
        return self.do_srv2clnt_grad
    def reset_do_grad(self, reset):
        self.do_srv2clnt_grad = reset

    def srv2clnt_grad(self, server_grads):
        self.net.train() ; self.net = self.net.to(self.device)
        # optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr)
        self.optimizer_client = torch.optim.SGD(self.net.parameters(), lr = self.lr, weight_decay = 1e-3, momentum = 0.9) #0.9 client_momentum = 0 比较好，但前期差异也没有很大
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_client, step_size = 1, gamma = 1)
        # self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)

        for fx, AN_grad, server_grad in \
            zip(self.fxs, self.AN_grads, server_grads):
            self.optimizer_client.zero_grad()
            fx.backward(server_grad)
            torch.nn.utils.clip_grad_norm_(parameters=self.net.parameters(), max_norm=1)  # 10 #5,3
            self.optimizer_client.step()
        self.fxs = []
        self.AN_grads = []
        # Freeze model
        self.scheduler.step()
        for params in self.net.parameters():
            params.requires_grad = False

        self.net.eval()

    def train(self, net):
        self.net = net
        self.ldr_train = DataLoader(Dataset(self.dataset_train,self.idxs,train=True, dataset_name = self.dataset_name),batch_size=self.bs,shuffle=True)
        self.net.train() ; self.net = self.net.to(self.device)

        # optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        self.optimizer_client = torch.optim.SGD(self.net.parameters(), lr = self.lr, weight_decay = 1e-3, momentum = 0.9) #0.9 client_momentum = 0 比较好，但前期差异也没有很大
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_client, step_size = 1, gamma = 1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_client, T_max=5)
        #----- core training--------
        self.net.train()
        smashed_list = []
        trn_gene_iter = self.ldr_train.__iter__()
        for iter in range(self.local_ep): #以batch_zize为一次迭代
            images, labels = trn_gene_iter.__next__()
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer_client.zero_grad()
            #---------forward prop-------------
            # if self.need_ffc:images = self.FFC_C2(images)

            fx = self.net(images)
            smashed_tmp = fx.clone().detach().requires_grad_(True)               
            smashed = copy.deepcopy(smashed_tmp)
            local_labels = copy.deepcopy(labels)
            smashed_list.append(list((smashed,local_labels)))
            #TODO:添加一个辅助网络的

            #----- Update with AN 
            #----- Sending activations to AN and receiving gradients from AN
            [smashed_data_AN, self.AN_net] = self.train_AN(smashed,labels, self.AN_net)
            if self.do_srv2clnt_grad:
                self.fxs.append(fx)
                self.AN_grads.append(smashed_data_AN)
                #---client local update---
                fx.backward(smashed_data_AN)
                #TODO:验证一下local model是否update了
                torch.nn.utils.clip_grad_norm_(parameters=self.net.parameters(), max_norm= 1)#10 #5,3
                self.optimizer_client.step()
                                
                # print("'Client ID: %.3d', 'loc_ep: %.3d','Client LR: %.4f'" %(self.idx, iter, scheduler.get_lr()[0]))
        if self.do_srv2clnt_grad:
            self.scheduler.step()

        #Freeze model
        for params in self.net.parameters():
            params.requires_grad = False

        self.net.eval()

        return self.net.state_dict(), self.AN_net.state_dict(), smashed_list

    def train_AN(self, smashed_data, labels, AN_net):
        self.AN_net = AN_net
        net_AN = copy.deepcopy(self.AN_net).to(self.device)
        # optimizer_AN = torch.optim.Adam(net_AN.parameters(), lr = self.lr) 
        optimizer_AN = torch.optim.SGD(net_AN.parameters(), lr = self.lr, weight_decay = 1e-3, momentum = 0.9)#0.9
        loss_AN_fn = torch.nn.CrossEntropyLoss(reduction='sum')


        #TODO:缓存清除
        #--train and update for AN ----
        net_AN.train()
        optimizer_AN.zero_grad()
        smashed_data = smashed_data.to(self.device)
        labels = labels.to(self.device)
        # file = open('output.txt', 'w+')
        # sys.stdout = file
        # print(f'before train: {smashed_data.grad}')
        #---forward prop----
        fx_AN = net_AN(smashed_data)
        # print(f'after train: {smashed_data.grad}')


        #---calculate loss---
        loss_AN = loss_AN_fn(fx_AN+1e-8, labels.reshape(-1).long())
        loss_AN = loss_AN/list(labels.size())[0]
        assert torch.isnan(loss_AN).sum() == 0, print('Error:loss_AN.sum()=0!',loss_AN)
        # acc_AN = calculate_acc(fx_AN,labels)
        #TODO：用户的train怎么print出来

        #---backward prop----
        loss_AN.backward()
        smashed_data_AN = smashed_data.grad.clone().detach() #计算了张量的梯度和副本
        # print(f'after backward back shape: {smashed_data_AN.shape}')
        # print(f'after backward, back: {smashed_data_AN}')
        # print(f'after backward smashed_data.grad shape: {smashed_data.grad.shape}')
        # print(f'after backward, smashed_data.grad: {smashed_data.grad}')
        # print(f'sub: {smashed_data_AN-smashed_data.grad}')
        # file.close()
        # exit(0)

        torch.nn.utils.clip_grad_norm_(parameters=net_AN.parameters(), max_norm=1)#10,5,3
        optimizer_AN.step()
        # print("'LOCAL AN LR: %.4f'" %(self.lr))
        
        
        #Freeze model
        for params in net_AN.parameters():
            params.requires_grad = False
        net_AN.eval()
        
        # self.AN_loss_train.append(loss_AN.item())
        # self.AN_acc_train.append(acc_AN.item())

        #---Update AN model for clnt---
        AN_net_update = copy.deepcopy(net_AN)

        return smashed_data_AN, AN_net_update
