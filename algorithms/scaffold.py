import torch

class SCAFFOLD:
    
    def __init__(self, clients, server, algorithm_type="SCAFFOLD"):
        

        self.clients = clients
        self.server = server
        self.algorithm_type = algorithm_type
        
        server_params = self.server.get_params()
        self.server_control = {
            "flat_params": torch.zeros_like(server_params["flat_params"]),
            "shapes": server_params["shapes"],
            "names": server_params["names"]
        }
        
        self.client_controls = []
        for client in self.clients:
            client_control = {
                "flat_params": torch.zeros_like(server_params["flat_params"]),
                "shapes": server_params["shapes"],
                "names": server_params["names"]
            }
            self.client_controls.append(client_control)

    def scaffold(self, iteration, criterion, gamma, inner_iter=1, record_interval=2):

        params = self.server.get_params()
        device = self.server.device
        loss = []
        accuracy = []
        
        for t in range(iteration):

            selected_clients = self.clients  
            
            server_params = self.server.clone_params(params)
            server_control_flat = self.server_control["flat_params"].to(device)
            
            client_updates = []
            client_deltas = []  
            
            for i, client in enumerate(selected_clients):

                client_params = client.clone_params(server_params)
                client.set_params(client_params)
                
                client_control = self.client_controls[i]["flat_params"].to(client.device)
                
                initial_params = client.get_params()["flat_params"].clone()
                
                self._client_local_update(
                    client, 
                    criterion, 
                    inner_iter, 
                    server_control_flat.to(client.device),
                    client_control
                )
                
                final_params = client.get_params()["flat_params"]
                
                param_delta = (initial_params - final_params) / (client.optimizer.param_groups[0]['lr'] * inner_iter)
                
                control_delta = param_delta - server_control_flat.to(client.device)

                
                client_updates.append(param_delta.to(device))
                client_deltas.append(control_delta.to(device))
                
                self.client_controls[i]["flat_params"] = (
                    self.client_controls[i]["flat_params"] + control_delta.to(device)
                )
            
            avg_update = sum(client_updates) / len(client_updates)
            avg_control_delta = sum(client_deltas) / len(client_deltas)
            
            params["flat_params"] = params["flat_params"] - gamma * avg_update
            

            self.server_control["flat_params"] = (
                self.server_control["flat_params"] + 
                (len(selected_clients) / len(self.clients)) * avg_control_delta
            )
            
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
                print(f"SCAFFOLD Round {t}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%")
        
        return loss, accuracy
    
    def _client_local_update(self, client, criterion, inner_iter, server_control, client_control):

        client.model.train()
        
        for _ in range(inner_iter):
            for batch_idx, (data, target) in enumerate(client.dataloader):
                client.optimizer.zero_grad()
                data, target = data.to(client.device), target.to(client.device)
                output = client.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                param_idx = 0
                for param in client.model.parameters():
                    if param.grad is not None:
                        param_size = param.grad.numel()
                        server_ctrl_part = server_control[param_idx:param_idx + param_size].view_as(param.grad)
                        client_ctrl_part = client_control[param_idx:param_idx + param_size].view_as(param.grad)
                        
                        corrected_grad = param.grad + server_ctrl_part - client_ctrl_part
                        param.grad.copy_(corrected_grad)
                        
                        param_idx += param_size
                
                client.optimizer.step()
                break