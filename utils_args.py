import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    #Data Split 
    parser.add_argument('--iid', type = str, default = 'iid', help = '[iid, Noniid] where 0 means non-iid and 1 means iid')
    parser.add_argument('--dataset', type = str, default = 'CIFAR10', help = '{mnist, CIFAR10, cifar100}')

    parser.add_argument('--Group', type = int, default = 2, help = 'spliting type : 1 -- all C4; 2-- C2 and C4')
    parser.add_argument('--frac', type = float, default = 0.5, help = 'frac for G4')

    parser.add_argument('--K', type = float, default = 1, help = 'async K')
    parser.add_argument('--async_lr', type = float, default = 0.02, help =' the learning rate for async K')

    #Model Argument
    parser.add_argument('--model', type = str, default = 'AlexNet', help = '{LeNet5, MLP, AlexNet,...}')
    
    #Scenerio Argument    
    parser.add_argument('--num_users', type = int, default = 20, help = 'the number of clients')
    parser.add_argument('--global_iters', type = int, default = 50, help = 'global rounds/epochs ')
    # parser.add_argument('--global_lr', type = float, default = 0.1, help = 'global learning rate for aggregation')
    parser.add_argument('--local_lr', type = float, default = 0.001, help = 'the learning rate for both users and server net training')
    parser.add_argument('--local_bs', type = int, default = 32, help = 'local batch size for once training')
    parser.add_argument('--local_ep', type = int, default = 64, help = 'loal epoch each training')#60
    parser.add_argument('--local_Epoch',type = int, default = 5, help ='local Epochs')
    # parser.add_argument('--bs', type = float, default = 2000, help = 'test batch size')
    # parser.add_argument('--momentum', type = float, default = 0, help = 'the local SGD training momentum')设置了momentum = 0



# max_norm = 1
    args = parser.parse_args()
    return args

