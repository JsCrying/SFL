import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

import random
import numpy as np
import os

#%%--------------------
# Auxiliary Network
# ---------------------
class ANet_same(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super(ANet_same,self).__init__()
        dim_1: int = 64 * 16 * 16
        self.layer_1 = nn.Linear(dim_1, num_classes)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.layer_1(x)
        return x
    
class ANet_diff(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super(ANet_diff,self).__init__()
        dim_1: int = 128 * 7 * 7
        self.layer_1 = nn.Linear(dim_1, num_classes)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.layer_1(x)
        return x

class VGG16_server_same(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super(VGG16_server_same, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class VGG16_client_same(nn.Module):
    def __init__(self) -> None:
        super(VGG16_client_same, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        return x

class VGG16_server_diff(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super(VGG16_server_diff, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class VGG16_client_diff_1(nn.Module):
    def __init__(self) -> None:
        super(VGG16_client_diff_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x

class VGG16_client_diff_2(nn.Module):
    def __init__(self) -> None:
        super(VGG16_client_diff_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x    

#---------------------
# class AlexNet_server_same(nn.Module):
#     def __init__(self, num_classes: int = 10, dropout: float = 0.5):
#         super(AlexNet_server_same,self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.AdaptiveAvgPool2d((6, 6)),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout),
#             nn.Conv2d(256, 192, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Linear(192 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x  
#---------------------
# class AlexNet_client_same(nn.Module):
#     def __init__(self):
#         super(AlexNet_client_same,self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         return x   # output [10,9216]
#---------------------
# class AlexNet_client_diff_1(nn.Module):
#     def __init__(self):
#         super(AlexNet_client_diff_1, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 384, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),  #TODO：混合切的话，如果是切3层和切2层这里也得改一下
#             )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         # self.ffc = nn.Linear(3*32*32,192)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x = FFC_C2(x)
#         x = self.features(x)#输出[10,256,32,32]
#         x = self.avgpool(x)
#         return x   # output [10,9216]        

def load_pretrained(net, layer, part: str = 'features'):
    model = models.vgg16(pretrained=True)
    state_dict = model.state_dict()

    def wrap(k, layer):  # 名称一致: 在预训练这里是8: 在自己这里是3
        k = k.split('.')
        k[1] = str(layer[int(k[1])])
        return '.'.join(k)
    pretrained_dict = {
        wrap(k, layer): v for k, v in state_dict.items() if part in k
                                               and int(k.split('.')[1]) in layer.keys()
    }
    net_dict = net.state_dict()
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

    ''' 
        Client:
        features_1_4: [8]
        features_1_2_3: [0, 3, 6]
        features_1_2_3_4: [0, 3, 6, 8]
        Server:
        features_5: [10], classifier: [1, 4]
        features_4_5: [8, 10], classifier: [1, 4]
        最后一个线性层输出通道不同，因此不用预训练
    '''
