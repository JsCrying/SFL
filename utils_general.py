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
        file_tmp = 'cifar/40_iid_cifar_async_K/'
        if (not os.path.exists('save/%s' %(file_tmp))):
            os.makedirs('save/%s' %(file_tmp))
    elif rule_iid == 'Noniid':
        file_tmp = 'cifar/40_Noniid_cifar_async_K/'
        if (not os.path.exists('save/%s' %(file_tmp))):
            os.makedirs('save/%s' %(file_tmp))
    file_name = 'save/' + file_tmp + 'SFL_acc_' + str(args.K) + '_' + str(args.init_lr)
    write_info_to_txt(FL_acc, file_name)
    file_name = 'save/' + file_tmp + 'SFL_loss_' + str(args.K) + '_' + str(args.init_lr)
    write_info_to_txt(FL_loss, file_name)
    file_name = 'save/' + file_tmp + 'SFL_acc_trn_' + str(args.K) + '_' + str(args.init_lr)
    write_info_to_txt(FL_acc_trn, file_name)
    file_name = 'save/' + file_tmp + 'SFL_loss_trn_' + str(args.K) + '_' + str(args.init_lr)
    write_info_to_txt(FL_loss_trn, file_name)