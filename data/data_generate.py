import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_mnist_dataset(download=True):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  
        transforms.Lambda(lambda x: x.view(-1)) 
    ])
    
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=download, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', 
        train=False, 
        transform=transform
    )
    
    return train_dataset, test_dataset

def get_mnist_binary_dataset(class_a=0, class_b=1, download=True):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  
    ])
    
    full_train = datasets.MNIST('./data', train=True, download=download, transform=transform)
    full_test = datasets.MNIST('./data', train=False, transform=transform)
    
    train_indices = [i for i, (_, label) in enumerate(full_train) if label in [class_a, class_b]]
    test_indices = [i for i, (_, label) in enumerate(full_test) if label in [class_a, class_b]]
    

    train_dataset = Subset(full_train, train_indices)
    test_dataset = Subset(full_test, test_indices)
    
    def relabel_subset(dataset, class_a, class_b):
        new_targets = []
        for _, label in dataset:
            new_label = 0 if label == class_a else 1
            new_targets.append(new_label)
        return new_targets
    
    train_targets = relabel_subset(train_dataset, class_a, class_b)
    test_targets = relabel_subset(test_dataset, class_a, class_b)
    
    class RelabeledDataset(torch.utils.data.Dataset):
        def __init__(self, subset, new_targets):
            self.subset = subset
            self.new_targets = new_targets
            
        def __getitem__(self, index):
            data, _ = self.subset[index]
            return data, self.new_targets[index]
            
        def __len__(self):
            return len(self.subset)
    
    train_dataset = RelabeledDataset(train_dataset, train_targets)
    test_dataset = RelabeledDataset(test_dataset, test_targets)
    
    return train_dataset, test_dataset

def get_cifar10_dataset(download=True):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        './data/CIFAR-10', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        './data/CIFAR-10', 
        train=False, 
        transform=transform_test
    )
    
    return train_dataset, test_dataset

