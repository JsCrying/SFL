from torch.utils.data import DataLoader
from utils_dataset import Dataset
import torch
import copy
from .server_side_SL import *
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
    def __init__(self, args, clnt_model, idx, device, net_server,
                 dataset_train = None, dataset_test = None, idxs = None, idxs_test = None, dataset_name = 'CIFAR10'):
        self.args = args
        self.clnt_model = clnt_model
        self.idx = idx
        self.lr = args.local_lr / args.diff_lr # 差异化与server LR
        self.lr_server = args.local_lr
        self.local_ep = args.local_ep
        self.device = device
        self.dataset_train = dataset_train
        self.idxs = idxs # tain labels
        
        self.dataset_test = dataset_test
        self.idxs_test = idxs_test
        self.dataset_name = dataset_name

        self.bs = args.local_bs
        self.dataset_name = args.dataset

        self.net_server = net_server

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
        trn_gene_iter = self.ldr_train.__iter__()
        for iter in range(self.local_ep): #以batch_zize为一次迭代
            images, labels = trn_gene_iter.__next__()
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer_client.zero_grad()
            #---------forward prop-------------
            fx = self.net(images)
            smashed_tmp = fx.clone().detach().requires_grad_(True)               
            smashed = copy.deepcopy(smashed_tmp)
            local_labels = copy.deepcopy(labels)

            #----- Update with server
            #----- Sending activations to server and receiving gradients from server
            [net_server_update, dfx_client] = train_server(self.net_server, smashed, local_labels, self.device, self.lr_server)

            #---client local update---
            fx.backward(dfx_client)
            torch.nn.utils.clip_grad_norm_(parameters=self.net.parameters(), max_norm= 1)#10 #5,3
            self.optimizer_client.step()                  

        self.scheduler.step()

        #Freeze model
        for params in self.net.parameters():
            params.requires_grad = False

        self.net.eval()

        return self.net.state_dict(), net_server_update