import torch
from torch.utils.data import DataLoader

class Node:
    def __init__(self,model):
        self.model = model
    
    def get_params(self):
        
        params = []
        shapes = []
        names = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param.data.view(-1))
                shapes.append(param.shape)
                names.append(name)

        flat_params = torch.cat(params)
        return {
            "flat_params": flat_params,
            "shapes": shapes,
            "names": names
        }
    
    def set_params(self, params):
        flat_params = params["flat_params"]
        shapes = params["shapes"]
        names = params["names"]
        
        param_dict = {}
        start_idx = 0
        
        for name, shape in zip(names, shapes):
            num_params = int(torch.prod(torch.tensor(shape)))
            end_idx = start_idx + num_params
            param_dict[name] = flat_params[start_idx:end_idx].view(shape)
            start_idx = end_idx
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in param_dict:
                    param.data.copy_(param_dict[name])

    def clone_params(self, params):
        cloned_flat_params = params["flat_params"].clone()
        cloned_shapes = params["shapes"][:] 
        cloned_names = params["names"][:]   
        
        return {
            "flat_params": cloned_flat_params,
            "shapes": cloned_shapes,
            "names": cloned_names
        }
    
    

class Client(Node):
    def __init__(self, rank, model, dataloader, optimizer, device, name=None):
        super().__init__(model)
        self.rank = rank
        self.dataloader = dataloader
        self.name = name
        self.optimizer = optimizer
        self.device = device 
        self.model = self.model.to(device)
        
        

    def train(self, criterion, num_epoch=1, noise_std=0.01):
        self.model.train()
        
        for _ in range(num_epoch):
            
            for batch_idx, (data, target) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                # print(target)

                for param in self.model.parameters():
                    noise = torch.randn_like(param.data) * noise_std * self.optimizer.param_groups[0]['lr']
                    param.data.add_(noise)
                break



class Server(Node):
    def __init__(self, model, train_dataloader, test_dataloader, device):
        super().__init__(model)
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.model = self.model.to(device)
        self.device = device