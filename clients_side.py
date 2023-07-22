from torch.utils.data import DataLoader
from utils_dataset import Dataset
import torch
import copy
from server_side import *
from torch.utils import data

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
    def __init__(self, args, clnt_model, idx, device, AN_net, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None, dataset_name = 'CIFAR10'):
        args = args
        self.clnt_model = clnt_model
        self.idx = idx
        self.lr = args.local_lr
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
        
        

    def train(self, net):
        self.ldr_train = DataLoader(Dataset(self.dataset_train,self.idxs,train=True, dataset_name = self.dataset_name),batch_size=self.bs,shuffle=True)
        net.train() ; net = net.to(self.device)

        # optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        optimizer_client = torch.optim.SGD(net.parameters(), lr = self.lr, weight_decay = 1e-3, momentum = 0.9) #0.9 client_momentum = 0 比较好，但前期差异也没有很大
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_client, step_size = 1, gamma = 1)        

        #----- core training--------
        net.train()
        trn_gene_iter = self.ldr_train.__iter__()
        smashed_list = []
            
        #改成了local Epoch
        # Epoch = 5
        # for e in range(Epoch):
            # trn_gene_iter = self.ldr_train.__iter__()
            # smashed_list = []
        for iter in range(self.local_ep): #以batch_zize为一次迭代
            images,labels = trn_gene_iter.__next__()
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer_client.zero_grad()
            #---------forward prop-------------
            # if self.need_ffc:images = self.FFC_C2(images)

            fx = net(images)
            smashed_tmp = fx.clone().detach().requires_grad_(True)               
            smashed = copy.deepcopy(smashed_tmp)
            local_labels = copy.deepcopy(labels)
            smashed_list.append(list((smashed,local_labels)))
            #TODO:添加一个辅助网络的

            #----- Update with AN 
            #----- Sending activations to AN and receiving gradients from AN
            [smashed_data_AN, self.AN_net] = self.train_AN(smashed,labels, self.AN_net)
            
            #---client local update---
            fx.backward(smashed_data_AN)

            #TODO:验证一下local model是否update了
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm= 3)#10 #5,3
            optimizer_client.step()
                            
            # print("'Client ID: %.3d', 'loc_ep: %.3d','Client LR: %.4f'" %(self.idx, iter, scheduler.get_lr()[0]))                
        scheduler.step()

        #Freeze model
        for params in net.parameters():
            params.requires_grad = False

        net.eval()
        # return net.state_dict(), self.AN_net.state_dict(), smashed_data, local_labels, self.AN_loss_train, self.AN_acc_train
        return net.state_dict(), self.AN_net.state_dict(), smashed_list, self.AN_loss_train, self.AN_acc_train


    # def evaluate(self, net, ell, dataset_tst, dataset_tst_label, net_server):

    #     acc_overall = 0; loss_overall = 0;
    #     # loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        
    #     # batch_size = min(6000, dataset_tst.shape[0])
    #     # batch_size = min(2000, dataset_tst.shape[0])#选择batch_size小于2000
    #     batch_size = min(1000, dataset_tst.shape[0])#选择batch_size小于2000
    #     self.ldr_test = DataLoader(Dataset(dataset_tst, dataset_tst_label,train=False, dataset_name = self.dataset_name),
    #                     batch_size=batch_size,shuffle=False)

    #    #TODO:test bs可以传输args.bs进来
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

#TODO:测试ACC
    

    def FFC_C2(self,images):
        net_FFC = copy.deepcopy(self.FFC_net).to(self.device)
        net_FFC.train()
        fx_ffc = net_FFC(images)
        return fx_ffc

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

        #---forward prop----
        fx_AN = net_AN(smashed_data)

        #---calculate loss---
        loss_AN = loss_AN_fn(fx_AN+1e-8, labels.reshape(-1).long())
        loss_AN = loss_AN/list(labels.size())[0]
        assert torch.isnan(loss_AN).sum() == 0, print('Error:loss_AN.sum()=0!',loss_AN)
        # acc_AN = calculate_acc(fx_AN,labels)
        #TODO：用户的train怎么print出来

        #---backward prop----
        loss_AN.backward()
        smashed_data_AN = smashed_data.grad.clone().detach() #计算了张量的梯度和副本

        torch.nn.utils.clip_grad_norm_(parameters=net_AN.parameters(), max_norm=3)#10,5,3
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


def calculate_acc(fx_AN, labels):
    preds = fx_AN.max(1, keepdim = True)[1]
    correct = preds.eq(labels.view_as(preds)).sum()
    acc = 100.00 * correct.float()/preds.shape[0]
    return acc


# # --- Evaluate client_model + AN_model
# def get_acc_loss(data_x, data_y, client_model, AN_model, dataset_name, device, w_decay = None, ):
    
#     acc_overall = 0; loss_overall = 0;
#     loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
#     batch_size = min(6000, data_x.shape[0])
#     batch_size = min(2000, data_x.shape[0])#选择batch_size小于2000
#     n_tst = data_x.shape[0]
#     tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)#Train test的时候shuffle=False
#     model.eval(); model = model.to(device)
#     with torch.no_grad():
#         tst_gen_iter = tst_gen.__iter__()
#         for i in range(int(np.ceil(n_tst/batch_size))):#ceil向上取整
#             batch_x, batch_y = tst_gen_iter.__next__()#为什么会跳到_getitem__()
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
#             y_pred = model(batch_x)
            
#             loss = loss_fn(y_pred, batch_y.reshape(-1).long())

#             loss_overall += loss.item()

#             # Accuracy calculation
#             y_pred = y_pred.cpu().numpy()            
#             y_pred = np.argmax(y_pred, axis=1).reshape(-1)
#             batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
#             batch_correct = np.sum(y_pred == batch_y)
#             acc_overall += batch_correct
    
    
#     loss_overall /= n_tst
#     if w_decay != None:
#         # Add L2 loss
#         params = get_mdl_params([model], n_par=None)
#         loss_overall += w_decay/2 * np.sum(params * params)
        
#     model.train()#重启？
#     return loss_overall, acc_overall / n_tst   


# def get_mdl_params(model_list, n_par=None):
    
#     if n_par==None:
#         exp_mdl = model_list[0]
#         n_par = 0 #参数个数？199210
#         for name, param in exp_mdl.named_parameters():#named_parameters() vs parameters()前者给出网络层的名字和参数的迭代器，后者给出参数迭代器
#             n_par += len(param.data.reshape(-1))
    
#     param_mat = np.zeros((len(model_list), n_par)).astype('float32')
#     for i, mdl in enumerate(model_list):
#         idx = 0
#         for name, param in mdl.named_parameters():
#             temp = param.data.cpu().numpy().reshape(-1) #为什么要把数据cpu()
#             param_mat[i, idx:idx + len(temp)] = temp
#             idx += len(temp)
#     return np.copy(param_mat)


