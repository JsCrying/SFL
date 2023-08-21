from torch.utils.data import DataLoader
from utils_dataset import Dataset
import torch
import copy
from .server_side_FL import *
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
    def __init__(self, args, idx, device, recv_iter=0,
                 dataset_train = None, dataset_test = None, idxs = None, idxs_test = None, dataset_name = 'CIFAR10'):
        self.args = args
        self.idx = idx
        self.lr = args.local_lr
        self.local_ep = args.local_ep
        self.device = device
        self.dataset_train = dataset_train
        self.idxs = idxs # tain labels
        
        self.dataset_test = dataset_test
        self.idxs_test = idxs_test
        self.dataset_name = dataset_name

        self.bs = args.local_bs
        self.dataset_name = args.dataset

        self.recv_iter = recv_iter # 收到来自server模型的服务器周期

    def train(self, net):
        self.net = net
        self.ldr_train = DataLoader(Dataset(self.dataset_train,self.idxs,train=True, dataset_name = self.dataset_name),batch_size=self.bs,shuffle=True)
        self.net.train() ; self.net = self.net.to(self.device)

        # optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        self.optimizer_client = torch.optim.SGD(self.net.parameters(), lr = self.lr, weight_decay = 1e-3, momentum = 0.9) #0.9 client_momentum = 0 比较好，但前期差异也没有很大
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_client, step_size = 1, gamma = 1)
        self.loss_AN_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_client, T_max=5)
        #----- core training--------
        self.net.train()

        trn_gene_iter = self.ldr_train.__iter__()
        for iter in range(self.local_ep): #以batch_zize为一次迭代
            images, labels = trn_gene_iter.__next__()
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer_client.zero_grad()
            #---------forward prop-------------
            # if self.need_ffc:images = self.FFC_C2(images)

            fx = self.net(images)

            #---calculate loss---
            loss_AN = self.loss_AN_fn(fx+1e-8, labels.reshape(-1).long())
            loss_AN = loss_AN/list(labels.size())[0]

            #---client local update---
            loss_AN.backward()
            #TODO:验证一下local model是否update了
            torch.nn.utils.clip_grad_norm_(parameters=self.net.parameters(), max_norm= 1)#10 #5,3
            self.optimizer_client.step()
                                
                # print("'Client ID: %.3d', 'loc_ep: %.3d','Client LR: %.4f'" %(self.idx, iter, scheduler.get_lr()[0]))

        self.scheduler.step()

        #Freeze model
        for params in self.net.parameters():
            params.requires_grad = False

        self.net.eval()

        return self.net.state_dict()
