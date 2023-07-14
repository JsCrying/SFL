import random
import torch
import numpy as np
import json

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