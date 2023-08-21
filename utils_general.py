import random
import torch
import numpy as np
import json
import os

def random_seed(SEED=37):
    SEED = 37
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(torch.cuda.get_device_name(0))  


# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 

def write_info_to_txt(info,name):
    with open("{}.txt".format(name), 'w') as f:
        json.dump(info, f)

def record(FL_acc, FL_loss, FL_acc_trn, FL_loss_trn, args, rule_iid):
    #%----------------------记录---------------------------------------------------------------
    if rule_iid == 'iid':
        file_tmp = 'cifar/20_iid_cifar_async_K/'
        if (not os.path.exists('save/%s' %(file_tmp))):
            os.makedirs('save/%s' %(file_tmp))
    elif rule_iid == 'Noniid':
        file_tmp = 'cifar/20_Noniid_cifar_async_K/'
        if (not os.path.exists('save/%s' %(file_tmp))):
            os.makedirs('save/%s' %(file_tmp))
    file_name = 'save/' + file_tmp + '_acc_' + str(args.K) + '_' + str(args.init_lr) + '_' + str(args.Group) + args.RUN
    write_info_to_txt(FL_acc, file_name)
    file_name = 'save/' + file_tmp + '_loss_' + str(args.K) + '_' + str(args.init_lr) + '_' + str(args.Group) + args.RUN
    write_info_to_txt(FL_loss, file_name)
    file_name = 'save/' + file_tmp + '_acc_trn_' + str(args.K) + '_' + str(args.init_lr) + '_' + str(args.Group) + args.RUN
    write_info_to_txt(FL_acc_trn, file_name)
    file_name = 'save/' + file_tmp + '_loss_trn_' + str(args.K) + '_' + str(args.init_lr) + '_' + str(args.Group) + args.RUN
    write_info_to_txt(FL_loss_trn, file_name)

import torch
from tqdm import tqdm


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    '''dataset: 0-1 range (ToTensor())'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in tqdm(dataloader):
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor