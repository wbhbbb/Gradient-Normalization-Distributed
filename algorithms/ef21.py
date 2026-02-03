import torch
from utils.compressor import TopK, RandK, uRandK, FCC, RandomQuantization,Identity

class EF21:

    SUPPORTED_ALGORITHMS = ["EF21-SGD", "EF21-SGDM"]
    SUPPORTED_COMPRESSORS = ["u_rand_k", "top_k", "rand_k","identity","rand_quant"]
    
    def __init__(self, clients, server, theta=0.0, algorithm_type="EF21-SGD", 
                 compressor_type="rand_k", compressor_param=0.1, compress_round = 1):
        
        self.clients = clients
        self.server = server
        self.theta = theta
        self.compressor_param = compressor_param
        self.R = compress_round
        self.algorithm_type = algorithm_type
        
        if algorithm_type not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm_type}. Supported: {self.SUPPORTED_ALGORITHMS}")
        self.algorithm_type = algorithm_type
        
        if compressor_type not in self.SUPPORTED_COMPRESSORS:
            raise ValueError(f"Unsupported compressor: {compressor_type}. Supported: {self.SUPPORTED_COMPRESSORS}")
        self.compressor_type = compressor_type
        
        self._init_compressor()
        

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
    
    def ef21(self, iteration, criterion, gamma, inner_iter=1, lr_decay = False, record_interval=2):
        import random
        seed=42
        random.seed(seed)
        torch.manual_seed(seed)
        
        params = self.server.get_params()
        device = self.server.device
        loss = []
        accuracy = []
        
        error_feedback = [self._params_to_dict(torch.zeros_like(params["flat_params"]), params["shapes"], params["names"]) 
                         for _ in range(len(self.clients))]
        
        momentum = [self._params_to_dict(torch.zeros_like(params["flat_params"]), params["shapes"], params["names"]) 
                     for _ in range(len(self.clients))]
        
        for t in range(iteration):
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

                update_inner_flat = (client_params["flat_params"] - params_new["flat_params"]) / (client.optimizer.param_groups[0]['lr'] * inner_iter)
                update_inner_dict = self._params_to_dict(update_inner_flat, params["shapes"], params["names"])

                for name in momentum[i].keys():
                    momentum[i][name] = self.theta * momentum[i][name] + (1 - self.theta) * update_inner_dict[name].to(device)

                update_dict = momentum[i]

                update_with_error_dict = {}
                
                for name in update_dict.keys():
                    update_with_error_dict[name] = update_dict[name] - error_feedback[i][name].to(device)

                compressed_update_dict = self.compressor.compress(update_with_error_dict)

                for name in compressed_update_dict.keys():
                    compressed_update_dict[name] = compressed_update_dict[name] + error_feedback[i][name].to(device)

                for name in error_feedback[i].keys():
                    error_feedback[i][name] = update_dict[name].clone()
                
                compressed_update_flat = self._dict_to_params(compressed_update_dict, params["shapes"], params["names"])
                compressed_updates.append(compressed_update_flat)

            
            avg_update = sum(compressed_updates) / len(self.clients)
            
            params["flat_params"] -= lr * avg_update
            self.server.set_params(params)
            
            if t % record_interval == 0:
                total_loss = 0.0
                total_samples = 0
                correct = 0
                
                self.server.model.eval()
                with torch.no_grad():
                    for inputs, labels in self.server.test_dataloader:
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
                print(f"{self.algorithm_type} Round {t}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%")
        
        return loss, accuracy
