import torch
from utils.compressor import *
from torch.utils.data import DataLoader

class QSGD:

    SUPPORTED_ALGORITHMS = ["QSGD", "QSGD-M", "QSGD-GN", "QSGD-GN-M"]
    SUPPORTED_COMPRESSORS = ["u_rand_k", "top_k", "rand_k","identity","rand_quant"]
    def __init__(self, clients, server, theta=0.0, algorithm_type="QSGD",
                 compressor_type="rand_k", compressor_param=0.1, compress_round = 1):
        self.clients = clients
        self.server = server
        self.theta = theta
        self.compressor_param = compressor_param
        self.R = compress_round

        if algorithm_type not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm_type}. Supported: {self.SUPPORTED_ALGORITHMS}")
        self.algorithm_type = algorithm_type

        if compressor_type not in self.SUPPORTED_COMPRESSORS:
            raise ValueError(f"Unsupported compressor: {compressor_type}. Supported: {self.SUPPORTED_COMPRESSORS}")
        self.compressor_type = compressor_type
        
        self._init_compressor()

        if algorithm_type == "QSGD":
            self.theta=0
        
        if algorithm_type in ["QSGD-GN-M", "QSGD-GN"]:
            self.with_gn = True
        else:
            self.with_gn = False        

    def _init_compressor(self):

        if self.compressor_type == "rand_k":
            compressor = RandK(ratio_or_k=self.compressor_param)
        elif self.compressor_type == "top_k":
            compressor = TopK(ratio_or_k=self.compressor_param)
        elif self.compressor_type == "u_rand_k":
            compressor = uRandK(ratio_or_k=self.compressor_param)
        elif self.compressor_type == "rand_quant":
            compressor = RandomQuantization()
        else:  
            compressor = Identity(ratio_or_k=self.compressor_param) 
        
        self.compressor = FCC(compressor, self.R)

    def _params_to_dict(self, flat_params, shapes, names):
        param_dict = {}
        start_idx = 0
        
        for name, shape in zip(names, shapes):
            num_params = int(torch.prod(torch.tensor(shape)))
            end_idx = start_idx + num_params
            param_tensor = flat_params[start_idx:end_idx].view(shape)
            param_dict[name] = param_tensor
            start_idx = end_idx
        
        return param_dict
    
    def _dict_to_params(self, param_dict, shapes, names):
        params = []
        for name, shape in zip(names, shapes):
            params.append(param_dict[name].view(-1))
        
        flat_params = torch.cat(params)
        return flat_params

    def count_nonzero_elements(self, tensor):
        return torch.count_nonzero(tensor).item()

    def qsgd(self, iteration, criterion, gamma, inner_iter=1, lr_decay = False, record_interval=1):
        import random
        seed=42
        random.seed(seed)
        torch.manual_seed(seed)
        params = self.server.get_params()
        device = self.server.device
        momentum = [self._params_to_dict(torch.zeros_like(params["flat_params"]), params["shapes"], params["names"]) 
                   for _ in range(len(self.clients))]
        loss = []
        accuracy = []
        grad_norm = []
        for t in range((iteration+self.R-1)//self.R):
            if lr_decay:
                lr = gamma/(t+1)**(1/2)
            else:
                lr = gamma
            compressed_updates = []
            for i, client in enumerate(self.clients):
                client_params = client.clone_params(params)
                client.set_params(client_params)
                client.train(criterion, num_epoch=inner_iter)

                params_new = client.get_params()

                update_inner_flat = (client_params["flat_params"]-params_new["flat_params"])/(client.optimizer.param_groups[0]['lr']*inner_iter)
                update_inner_dict = self._params_to_dict(update_inner_flat, params["shapes"], params["names"])

                for name in momentum[i].keys():
                    momentum[i][name] = self.theta * momentum[i][name] + (1-self.theta) * update_inner_dict[name].to(device)
            
                
                compressed_update_dict = self.compressor.compress(momentum[i])
                
                compressed_update_flat = self._dict_to_params(compressed_update_dict, params["shapes"], params["names"])
                compressed_updates.append(compressed_update_flat)
               
            update = sum(compressed_updates) / len(self.clients)
            
            if self.with_gn:
                norm = torch.norm(update,p=2)
                update = update/norm

            params["flat_params"]-= lr *update
            self.server.set_params(params)

            if t % record_interval == 0:
                total_loss = 0.0
                total_samples = 0
                correct = 0

                self.server.model.eval()
                with torch.no_grad():
                    for inputs,labels in self.server.test_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = self.server.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        _, predicted = torch.max(outputs, 1) 
                        correct += (predicted == labels).sum().item() 
            
                        batch_size = inputs.size(0)
                        total_loss += batch_loss.item() * batch_size
                        total_samples += batch_size

                avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                acc = 100 * correct / total_samples if total_samples > 0 else 0.0
                loss.append(avg_loss)
                accuracy.append(acc)
                if self.with_gn:
                    grad_norm.append(norm)
                    print(f"Norm = {norm}")
                else:
                    grad_norm.append(torch.norm(update,2))
                    print(f"Norm = {torch.norm(update,2)}")
                print(f"{self.algorithm_type} Round {t}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%")
                        
        return loss, accuracy, grad_norm
    

    def change_batch_size(self):
        for i, client in enumerate(self.clients):
            ori_dataloader = client.dataloader
            
            new_dataloader = DataLoader(
                dataset=ori_dataloader.dataset,
                batch_size=self.R * ori_dataloader.batch_size,
                shuffle=getattr(ori_dataloader, 'shuffle', False), 
                num_workers=getattr(ori_dataloader, 'num_workers', 0),
                collate_fn=getattr(ori_dataloader, 'collate_fn', None),
                pin_memory=getattr(ori_dataloader, 'pin_memory', False),
                drop_last=getattr(ori_dataloader, 'drop_last', False)
            )
            client.dataloader = new_dataloader
    

