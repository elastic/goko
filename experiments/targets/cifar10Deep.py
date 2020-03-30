import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from .datasets import CIFAR10
from .utils import *

class CIFAR10Deep7(nn.Module):
    def __init__(self):
        super(CIFAR10Deep7, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(32*32*3, 750),
            nn.ReLU(True),
            nn.AlphaDropout(p=0.25),
            nn.Linear(750, 750),
            nn.ReLU(True),
            nn.Linear(750, 750),
            nn.ReLU(True),
            nn.Linear(750, 750),
            nn.ReLU(True),
            nn.Linear(750, 750),
            nn.ReLU(True),
            nn.Linear(750, 150),
            nn.AlphaDropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(150, 10))
        self.add_module("5_layer",self.mlp)


    def forward(self, x):
        x = x.view(-1, 32*32*3)
        return self.mlp(x)

    def dataset(self):
            return CIFAR10

def trainCIFAR10Deep7(num_components=3,device=None,directory = ''):
    if device is None:
        device = getDevice()

    net = CIFAR10Deep()
    cifar = CIFAR10()
    batch_size = 120
    trainloader = cifar.training(batch_size)

    print('Training CIFAR10 Deep MLP Model')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    trainModel(net,trainloader,optimizer,criterion,400,device)
    
    net.eval()
    print('Finished Training, getting accuracy')
    testloader = cifar.testing()
    accuracy = testAccuracy(net,testloader)
    
    model_path = os.path.join(directory, "cifar10Deep7.pkl")
    print('Saving as: cifar10Deep7.pkl, with accuracy %.4f'%(accuracy,))
    torch.save(net,model_path)

class CIFAR10Deep10(nn.Module):
    def __init__(self):
        super(CIFAR10Deep10, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(32*32*3, 750),
            nn.ReLU(True),
            nn.AlphaDropout(p=0.25),
            nn.Linear(750, 750),
            nn.ReLU(True),
            nn.Linear(750, 750),
            nn.ReLU(True),
            nn.Linear(750, 750),
            nn.ReLU(True),
            nn.Linear(750, 750),
            nn.ReLU(True),
            nn.Linear(750, 150),
            nn.AlphaDropout(p=0.25),
            nn.ReLU(True),
            nn.Linear(150, 10))
        self.add_module("5_layer",self.mlp)


    def forward(self, x):
        x = x.view(-1, 32*32*3)
        return self.mlp(x)

    def dataset(self):
            return CIFAR10

def trainCIFAR10Deep10(num_components=3,device=None,directory = ''):
    if device is None:
        device = getDevice()

    net = CIFAR10Deep()
    cifar = CIFAR10()
    batch_size = 120
    trainloader = cifar.training(batch_size)

    print('Training CIFAR10 Deep MLP Model')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    trainModel(net,trainloader,optimizer,criterion,400,device)
    
    net.eval()
    print('Finished Training, getting accuracy')
    testloader = cifar.testing()
    accuracy = testAccuracy(net,testloader)
    
    model_path = os.path.join(directory, "cifar10Deep10.pkl")
    print('Saving as: cifar10Deep10.pkl, with accuracy %.4f'%(accuracy,))
    torch.save(net,model_path)