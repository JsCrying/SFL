import torch
from torch import nn
import torch.nn.functional as F

import random
import numpy as np
import os


#%%--------------------
#  Model at server side
# ---------------------

#-------CIFAR10 1层Conv+3FC ----------------------------------------
class AlexNet_server_side_C4(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super(AlexNet_server_side_C4,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  

#-------CIFAR10 CLIENTS GROUP1 4Convs -------------------------------
class AlexNet_client_side_C4(nn.Module):
    def __init__(self):
        super(AlexNet_client_side_C4,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x   # output [10,9216]

#-------CIFAR10 CLIENTS GROUP1 2Convs -------------------------------
class AlexNet_client_side_C2(nn.Module):
    def __init__(self):
        super(AlexNet_client_side_C2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 384, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),

            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),

            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  #TODO：混合切的话，如果是切3层和切2层这里也得改一下
            )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.ffc = nn.Linear(3*32*32,192)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = FFC_C2(x)
        x = self.features(x)#输出[10,256,32,32]
        x = self.avgpool(x)
        return x   # output [10,9216]        

#%%--------------------
# Auxiliary Network
# ---------------------
# 
class ANet_C4(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super(ANet_C4,self).__init__()
        # self.layer = nn.Linear(256*3*3,num_classes)#23050  
        # input_dim: int = 6912

        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        input_dim: int = 384 * 6 * 6
        self.layer_1 = nn.Linear(input_dim, 200)
        self.layer_2 = nn.Linear(200, num_classes)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x

# class ANet(nn.Module):
#     def __init__(self, num_classes: int = 10, dropout: float = 0.5):
#         super(ANet_C4,self).__init__()
#         # self.layer = nn.Linear(256*3*3,num_classes)#23050  
#         input_dim: int = 6912  
#         # input_dim: int = 256*6*6
#         self.layer = nn.Linear(input_dim,num_classes)

#     def forward(self,x:torch.Tensor) -> torch.Tensor:
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = self.layer(x)
#         return x


#------通道数预处理--------
class FFC_C2(nn.Module):
    def __init__(self):
        super(FFC_C2,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)#输出[10,256,32,32]
        return x   # output [10,9216]