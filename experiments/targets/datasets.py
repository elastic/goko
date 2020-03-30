import torch
import torchvision
from torchvision import transforms, utils, datasets
import torchvision.transforms as transforms

'''
To save space and bandwidth add a datasets_locations.py config file to this folder with:

mnist = "data"              # Put an absolute path to somewhere
cifar10 = "data"            # Put an absolute path to somewhere

imagenet_training = None    #Directory of Imagenet training
imagenet_testing = None     #Directory of Imagenet testing
'''
try:
    import datasets_locations
except ImportError:
    class dummie_class():
        def __init__(self):
            self.mnist = 'data'
            self.cifar10 = 'data'
            self.imagenet_training = None
            self.imagenet_testing = None
    datasets_locations = dummie_class()

def __transform_training__():
    transform = transforms.Compose(
        [transforms.RandomAffine(36, translate=(0.1,0.1), scale=(0.9,1.1), shear=20),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform

def __transform_testing__():
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform

def __transform_training_1d__():
    transform = transforms.Compose(
        [transforms.RandomAffine(36, translate=(0.1,0.1), scale=(0.9,1.1), shear=20),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    return transform

def __transform_testing_1d__():
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
    return transform

class DATASET():
    def __init__(self):
        raise NotImplementedError("Subclasses should implement this!")

    def training(self,batch_size = 120, num_workers = 48):
        self.trainloader = torch.utils.data.DataLoader(
                self.trainset, 
                batch_size=batch_size,
                shuffle=True, 
                pin_memory=True, 
                num_workers=num_workers,
                drop_last=True)
        return self.trainloader

    def testing(self,batch_size = 120, num_workers = 48):
        self.testloader = torch.utils.data.DataLoader(
            self.testset, 
            batch_size=batch_size,
            shuffle=False, 
            pin_memory=True, 
            num_workers=num_workers,
            drop_last=True)
        return self.testloader

    def getNames(self, arr):
        return [self.classes[i] for i in  arr]

class MNIST(DATASET):
    def __init__(self):
        self.trainset = torchvision.datasets.MNIST(root=datasets_locations.mnist, train=True,
                                    download=True, transform=__transform_training_1d__())

        self.testset = torchvision.datasets.MNIST(root=datasets_locations.mnist, train=False,
                                   download=True, transform=__transform_testing_1d__())

        self.classes = [str(i) for i in range(10)]

class CIFAR10(DATASET):
    def __init__(self):
        self.trainset = torchvision.datasets.CIFAR10(root=datasets_locations.cifar10, train=True,
                                    download=True, transform=__transform_training__())

        self.testset = torchvision.datasets.CIFAR10(root=datasets_locations.cifar10, train=False,
                                   download=True, transform=__transform_testing__())

        self.classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class IMAGENET(DATASET):
    def __init__(self):
        from imageNetClasses import imagenetDict
        transform = transforms.Compose(
            [transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1)),
            transforms.Resize((224,224)),
            transforms.ToTensor()])

        self.trainset = datasets.ImageFolder(
            datasets_locations.imagenet_training,
            transform=transform)

        self.testset = datasets.ImageFolder(
            datasets_locations.imagenet_testing, 
            transform=__transform_testing__())

        self.classes = imagenetDict