# code:utf-8  	Ubuntu
import sys
import os
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

import matplotlib.font_manager as mpt

#%% 多条曲线 for acc
file ='./save/10_Noniid/'
# file = file + '100epoch_fixedthrd/'
# file1 = file + 'SFL_acc_.txt'
# file2 = file + 'SFL_loss_.txt'
file3 = file + 'SFL_acc_trn_.txt'
# file4 = file + 'SFL_loss_trn_.txt'
# name1 = str('10users_FL(client,AN)_1server')
# name2 = str('10users_FL(client,AN)_1server')
name3 = str('10users_FL(client,AN)_1server_train')
# name4 = str('10users_FL(client,AN)_1server_train')
# name4 = str('exp_(0.4,0.1)')#存在解码误差
data = {}

# in_file = open(file1,mode='r')
# data_init = in_file.read()
# data_init = data_init[1:len(data_init)-1]
# data[name1] = list(map(float,data_init.split(",")))

# in_file = open(file2,mode='r')
# data_init = in_file.read()
# data_init = data_init[1:len(data_init)-1]
# data[name1] = list(map(float,data_init.split(",")))

in_file = open(file3,mode='r')
data_init = in_file.read()
data_init = data_init[1:len(data_init)-1]
data[name3] = list(map(float,data_init.split(",")))


# in_file = open(file4,mode='r')
# data_init = in_file.read()
# data_init = data_init[1:len(data_init)-1]
# data[name4] = list(map(float,data_init.split(",")))

colours = ['r','g','c','orange','violet','m','r','b','pink','k','y']

num_round = len(data[name3])
x = np.linspace(0,num_round,num_round)
i = 0
for key,value in data.items():
    line_name = key
    line_data = value
    colour = colours[i]  
    plt.plot(x,line_data,colour,label=line_name)
    i+=1


plt.grid(color="k", linestyle=":",axis='y')
plt.legend(loc="lower right")
plt.xlabel("Iteration")
# plt.title("Test Acc")
plt.title("Train Acc")
plt.ylabel("Accuracy")

# # plt.title("Test loss")
# plt.title("Train loss")
# plt.ylabel("Loss")

# timestamp= datetime.now().strftime("%Y%m%d%H%M%S")
figure = plt.gcf() # get current figure
figure.set_size_inches(19.2, 10.8)
plt.show(block=False)
plt.savefig('./save/SFL_trn_acc_10_clip_Noniid.svg',dpi = 100,bbox_inches='tight')
# plt.savefig('CIFAR10_NONIID03E5_updated_14.svg',dpi = 100,bbox_inches='tight')
# plt.savefig('CIFAR10_IID_updated_14.svg',dpi = 100,bbox_inches='tight')


print('wait')