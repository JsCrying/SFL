import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision import datasets, transforms
from datetime import datetime
from utils_general import *
from utils_args import *

args = args_parser()
args.K = 1
args.Group = 1
args.RUN = 'vgg16'

FL_loss = []
FL_acc = []
FL_loss_trn = []
FL_acc_trn = []

# dataset
# input_shape = 32
num_classes = 10

# hyper 
batch_size = 64
num_epochs = 500
learning_rate = 1e-3

# gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])
train_dataset = datasets.CIFAR10(root='../data/', 
                               download=True, 
                               train=True, 
                               transform=transform)
test_dataset = datasets.CIFAR10(root='../data/', 
                               download=True, 
                               train=False, 
                               transform=transform)


train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               shuffle=True, 
                                               batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                               shuffle=False, 
                                               batch_size=batch_size)

get_mean_and_std(train_dataset)

images, labels = next(iter(train_dataloader))

vgg = models.vgg16(pretrained=True)
in_features = vgg.classifier[6].in_features
vgg.classifier[6] = nn.Linear(in_features, 10)
vgg = vgg.to(device)

criterion = nn.CrossEntropyLoss()
# optimzier = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(vgg.parameters(), lr = learning_rate, momentum=0.9,weight_decay=5e-4)
total_batch = len(train_dataloader)

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # forward
        out = vgg(images)
        loss = criterion(out, labels)
        
        # 标准的处理，用 validate data；这个过程是监督训练过程，用于 early stop
        n_corrects = (out.argmax(axis=1) == labels).sum().item()
        acc = n_corrects/labels.size(0)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   # 更细 模型参数
        
        if (batch_idx+1) % 10 == 0:
            print(f'{datetime.now()}, {epoch+1}/{num_epochs}, {batch_idx+1}/{total_batch}: {loss.item():.4f}, acc: {acc}')

        if (batch_idx+1) % args.local_ep == 0:
            total = 0
            correct = 0
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                out = vgg(images)
                preds = torch.argmax(out, dim=1)
                
                total += images.size(0)
                correct += (preds == labels).sum().item()
            print(f'{correct}/{total}={correct/total}')
            FL_acc.append(correct/total*100.0)
        if (batch_idx+1) % 300 == 0:
            record(FL_acc, FL_loss, FL_acc_trn, FL_loss_trn, args, 'iid')
record(FL_acc, FL_loss, FL_acc_trn, FL_loss_trn, args, 'iid')