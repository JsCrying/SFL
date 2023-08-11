import os
import numpy as np
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms



#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this
def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
                          

class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, rule_arg, unbalanced_sgm=0,data_path =''):#similarity代替rule_arg
        self.dataset = dataset
        self.n_client = n_client
        self.seed = seed
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.rule = rule
        self.rule_arg = rule_arg
        self.instance_name = "%s_%d_%d_%s_%d" %(self.dataset,self.n_client,self.seed,self.rule,self.rule_arg) 
        self.set_data()
    
    def set_data(self):
        #Prepare data
        if not os.path.exists('%sData/%s' %(self.data_path,self.instance_name)):
            #%% Get Raw data
            if self.dataset == "mnist":
                print(self.dataset)
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trainset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                                                    train=True , download=True, transform=transform)
                testset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                                                    train=False, download=True, transform=transform)
                train_load = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=False, num_workers=1)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            if self.dataset == 'CIFAR10':
                print(self.dataset)
                transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

                trainset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                testset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                
                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=1)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            
            if self.dataset != 'emnist':
                train_itr = train_load.__iter__(); test_itr = test_load.__iter__() 
                # labels are of shape (n_data,)
                train_x, train_y = train_itr.__next__()
                test_x, test_y = test_itr.__next__()

                train_x = train_x.numpy(); train_y = train_y.numpy().reshape(-1,1)
                test_x = test_x.numpy(); test_y = test_y.numpy().reshape(-1,1)
            

            #%% Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(train_y))
            train_x = train_x[rand_perm]
            train_y = train_y[rand_perm]
            
            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y

            ###
            n_data_per_clnt = int((len(train_y)) / self.n_client)
            # Draw from lognormal distribution
            clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client))
            clnt_data_list = (clnt_data_list/np.sum(clnt_data_list)*len(train_y)).astype(int)
            diff = np.sum(clnt_data_list) - len(train_y)
            
            # Add/Subtract the excess number starting from first client
            if diff!= 0:
                for clnt_i in range(self.n_client):
                    if clnt_data_list[clnt_i] > diff:
                        clnt_data_list[clnt_i] -= diff
                        break

#%% Assign data
            if self.rule == 'Noniid' or self.rule == 'noniid':#'Drichlet'
                cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [np.where(train_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
    
                while(np.sum(clnt_data_list)!=0):
                    curr_clnt = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    print('Remaining Data: %d' %np.sum(clnt_data_list))
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if train_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        
                        clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = train_x[idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = train_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break
                
                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)
                
                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt,cls] = np.mean(clnt_y[clnt]==cls)
                prior_real_diff = np.abs(cls_means-cls_priors)
                print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))
            
            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm==0:
                assert len(train_y)//100 % self.n_client == 0 
                
                # create perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(train_y[:, 0])
                n_data_per_clnt = len(train_y) // self.n_client
                # clnt_x dtype needs to be float32, the same as weights
                clnt_x = np.zeros((self.n_client, n_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y = np.zeros((self.n_client, n_data_per_clnt, 1), dtype=np.float32)
                train_x = train_x[idx] # 50000*3*32*32
                train_y = train_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 100
                for i in range(self.n_client): # devices
                    for j in range(100): # class
                        clnt_x[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = train_x[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :, :, :]
                        clnt_y[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = train_y[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :] 
            

            elif self.rule == 'iid':#cifar100需要修改
               
                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
            
                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = train_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                    clnt_y[clnt_idx_] = train_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                               
                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)
           
            self.clnt_x = clnt_x; self.clnt_y = clnt_y
            self.test_x  = test_x;  self.test_y  = test_y
            
            # Save data
            os.mkdir('%sData/%s' %(self.data_path, self.instance_name))
            
            np.save('%sData/%s/clnt_x.npy' %(self.data_path, self.instance_name), clnt_x)
            np.save('%sData/%s/clnt_y.npy' %(self.data_path, self.instance_name), clnt_y)

            np.save('%sData/%s/test_x.npy'  %(self.data_path, self.instance_name),  test_x)
            np.save('%sData/%s/test_y.npy'  %(self.data_path, self.instance_name),  test_y)

#%% if data exists
        else:
            print("Data is already downloaded")
            self.clnt_x = np.load('%sData/%s/clnt_x.npy' %(self.data_path, self.instance_name),allow_pickle=True)
            self.clnt_y = np.load('%sData/%s/clnt_y.npy' %(self.data_path, self.instance_name),allow_pickle=True)
            self.n_client = len(self.clnt_x)

            self.test_x  = np.load('%sData/%s/test_x.npy'  %(self.data_path, self.instance_name),allow_pickle=True)
            self.test_y  = np.load('%sData/%s/test_y.npy'  %(self.data_path, self.instance_name),allow_pickle=True)
            
            if self.dataset == 'mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'fashion_mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'emnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;

        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " %clnt + 
                  ', '.join(["%.3f" %np.mean(self.clnt_y[clnt]==cls) for cls in range(self.n_cls)]) + 
                  ', Amount:%d' %self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]
               
        print('Total Amount:%d' %count)
        print('--------')

        print("      Test: " + 
              ', '.join(["%.3f" %np.mean(self.test_y==cls) for cls in range(self.n_cls)]) + 
              ', Amount:%d' %self.test_y.shape[0])  


#%% 
class Dataset(torch.utils.data.Dataset):    
    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist' :
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()
            
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
        
            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
                
        elif self.name == 'shakespeare':
            
            self.X_data = data_x
            self.y_data = data_y
                
            self.X_data = torch.tensor(self.X_data).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(self.y_data).float()     

    def __len__(self):#为什么？怎么用？在__next__()方法时会先调用__len__然后在调用__getitem__()
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist':
            X = self.X_data[idx, :]#[1,28,28]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y
        
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img # Horizontal flip
                if (np.random.rand() > .5):
                # Random cropping 
                    pad = 4
                    extended_img = np.zeros((3,32 + pad *2, 32 + pad *2)).astype(np.float32)
                    extended_img[:,pad:-pad,pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:,dim_1:dim_1+32,dim_2:dim_2+32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
            
        elif self.name == 'shakespeare':
            x = self.X_data[idx]
            y = self.y_data[idx] 
            return x, y
       