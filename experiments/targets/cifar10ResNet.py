import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
import os
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import os

from .datasets import CIFAR10
from .utils import *

class residual(nn.Module):
    def __init__(self,nxn,connections,padding):
        super(residual, self).__init__()
        self.conv1 = nn.Conv2d(connections, connections, nxn,padding = padding)
        self.bn1 = nn.BatchNorm2d(connections)
        self.conv2 = nn.Conv2d(connections, connections, nxn,padding = padding)
        self.bn2 = nn.BatchNorm2d(connections)
    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.bn1(y)
        y = F.relu(self.conv2(y))
        y = self.bn2(y)
        return x + y

class CIFAR10ResNet(nn.Module):
    def __init__(self,n):
        super(CIFAR10ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.res1 = []
        for i in range(0,n):
            res = residual(3,16,1)
            self.add_module("residual_"+str(i),res)
            self.res1.append(res)
        self.conv2 = nn.Conv2d(16, 32, 3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.res2 = []
        for i in range(n,2*n):
            res = residual(3,32,1)
            self.add_module("residual_"+str(i),res)
            self.res2.append(res)
        self.conv3 = nn.Conv2d(32, 64, 3,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.res3 = []
        for i in range(2*n,3*n):
            res = residual(3,64,1)
            self.add_module("residual_"+str(i),res)
            self.res3.append(res)

        self.fc1 = nn.Linear(64*8*8, 10)

        self.layerCount = n*2*3 + 4
    

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        for res in self.res1:
            x = res(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        for res in self.res2:
            x = res(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        for res in self.res3:
            x = res(x)

        x = x.view(-1, 64*8*8)
        x = self.fc1(x)
        return x

    def dataset(self):
        return CIFAR10

def trainCIFAR10ResNet(num_components=3,device=None,directory = ''):
    if device is None:
        device = getDevice()
        
    net = CIFAR10ResNet(num_components)
    cifar = CIFAR10()
    batch_size = 240
    trainloader = cifar.training(batch_size)
    

    print('Training CIFAR10 ResNet Model with %d layers'%(net.layerCount))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    trainModel(net,trainloader,optimizer,criterion,400)
    
    net.eval()
    print('Finished Training, getting accuracy')
    testloader = cifar.testing()
    accuracy = testAccuracy(net,testloader)

    model_path = os.path.join(directory, "cifar10ResNet%d.pkl"%(net.layerCount,))
    print('Saving as: cifar10ResNet%d.pkl, with accuracy %.4f'%(net.layerCount,accuracy,))
    torch.save(net,model_path)