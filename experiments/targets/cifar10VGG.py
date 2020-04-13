import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os

from .datasets import CIFAR10
from .utils import *

class CIFAR10VGG9(nn.Module):
    def __init__(self):
        super(CIFAR10VGG9, self).__init__()
        
        self.conv64 = nn.Sequential(
            nn.Conv2d(3, 64, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            )
        self.add_module("conv64",self.conv64)
        self.conv128 = nn.Sequential(
            nn.Conv2d(64, 128, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            )
        self.add_module("conv128",self.conv128)
        self.conv256 = nn.Sequential(
            nn.Conv2d(128, 256, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            )
        self.add_module("conv256",self.conv256)
        self.mlp = nn.Sequential(
            nn.AlphaDropout(p=0.25),
            nn.Linear(256*4*4, 256*4*4),
            nn.ReLU(True),
            nn.AlphaDropout(p=0.25),
            nn.Linear(256*4*4, 256*4*4),
            nn.ReLU(True),
            nn.Linear(256*4*4, 10))
        self.add_module("vgg",self.mlp)

        self.maxPooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv64(x)
        x = self.maxPooling(x)
        x = self.conv128(x)
        x = self.maxPooling(x)
        x = self.conv256(x)
        x = self.maxPooling(x)

        x = x.view(-1, 256*4*4)
        x = self.mlp(x)
        return x

def trainCIFAR10VGG9(device=None,directory = ''):
    if device is None:
        device = getDevice()
        
    net = CIFAR10VGG9()
    cifar = CIFAR10()
    batch_size = 240
    trainloader = cifar.training(batch_size)
    

    print('Training CIFAR10 VGG8 Model')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    trainModel(net,trainloader,optimizer,criterion,200)
    
    net.eval()
    print('Finished Training, getting accuracy')
    testloader = cifar.testing()
    accuracy = testAccuracy(net,testloader)

    model_path = os.path.join(directory, "cifar10VGG9.pkl")
    print('Saving as: cifarVGG9.pkl, with accuracy %.4f'%(accuracy,))
    torch.save(net,model_path)

class CIFAR10VGG12(nn.Module):
    def __init__(self):
        super(CIFAR10VGG12, self).__init__()
        
        self.conv64 = nn.Sequential(
            nn.Conv2d(3, 64, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            )
        self.add_module("conv64",self.conv64)
        self.conv128 = nn.Sequential(
            nn.Conv2d(64, 128, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            )
        self.add_module("conv128",self.conv128)
        self.conv256 = nn.Sequential(
            nn.Conv2d(128, 256, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            )
        self.add_module("conv256",self.conv256)
        self.mlp = nn.Sequential(
            nn.AlphaDropout(p=0.25),
            nn.Linear(256*4*4, 256*4*4),
            nn.ReLU(True),
            nn.AlphaDropout(p=0.25),
            nn.Linear(256*4*4, 256*4*4),
            nn.ReLU(True),
            nn.Linear(256*4*4, 10))
        self.add_module("vgg",self.mlp)

        self.maxPooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv64(x)
        x = self.maxPooling(x)
        x = self.conv128(x)
        x = self.maxPooling(x)
        x = self.conv256(x)
        x = self.maxPooling(x)

        x = x.view(-1, 256*4*4)
        x = self.mlp(x)
        return x

def trainCIFAR10VGG12(num_components=3,device=None,directory = ''):
    if device is None:
        device = getDevice()
        
    net = CIFAR10VGG12()
    cifar = CIFAR10()
    batch_size = 240
    trainloader = cifar.training(batch_size)
    

    print('Training CIFAR10 VGG12 Model')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    trainModel(net,trainloader,optimizer,criterion,200)
    
    net.eval()
    print('Finished Training, getting accuracy')
    testloader = cifar.testing()
    accuracy = testAccuracy(net,testloader)
    
    model_path = os.path.join(directory, "cifar10VGG12.pkl")
    print('Saving as: cifarVGG12.pkl, with accuracy %.4f'%(accuracy,))
    torch.save(net,model_path)