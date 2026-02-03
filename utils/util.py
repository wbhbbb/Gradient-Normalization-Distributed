import torch
from torch.utils.data import DataLoader
from model import *
from utils import Client, Server
import random

def create_fresh_models(model_type, dataset_list, train_dataloader, test_dataloader, batch_size, num_classes, inner_lr, client_device, server_device, algo_name, theta):

    seed=42
    random.seed(seed)
    torch.manual_seed(seed)
    num_clients = len(dataset_list)
    dataloaders = [DataLoader(dataset, batch_size, shuffle=True) for dataset in dataset_list]
    if model_type=="MLP":
        models = [MLP(input_size=784, num_classes=num_classes) for _ in range(num_clients)] 
        server_model = MLP(input_size=784, num_classes=num_classes)
    elif model_type=="ResNet-18":
        models = [ResNet() for _ in range(num_clients)]
        server_model = ResNet()
    if algo_name in ["FedAvg-M"]:
        theta = theta 
    else:
        theta = 0.0
    # print(algo_name,theta)
    optimizers = [torch.optim.SGD(model.parameters(), lr=inner_lr, momentum=theta, dampening=theta) for model in models]
    clients = [Client(i, models[i], dataloaders[i], test_dataloader, optimizers[i], client_device) for i in range(num_clients)]
    server = Server(server_model, train_dataloader, test_dataloader, server_device)
        
    return clients, server