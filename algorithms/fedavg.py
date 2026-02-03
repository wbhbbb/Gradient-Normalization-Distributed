import torch

class FedAvg:

    SUPPORTED_ALGORITHMS = ["FedAvg", "FedAvg-M", "FedAvg-GN", "FedAvg-GN-M"]
    def __init__(self,clients,server,theta,algorithm_type):
        self.clients = clients
        self.server = server
        self.algorithm_type = algorithm_type
        
        if algorithm_type == "FedAvg":
            self.with_gn = False
            self.theta = 0
        elif algorithm_type == "FedAvg-M":
            self.with_gn = False
            self.theta = 0
        elif algorithm_type == "FedAvg-GN":
            self.with_gn = True
            self.theta = 0
        elif algorithm_type == "FedAvg-GN-M":
            self.with_gn = True
            self.theta = theta
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_type}. Supported: {self.SUPPORTED_ALGORITHMS}")
        
    def get_grad(self,model):
        grad=[]
        for param in model.parameters():
            if param.requires_grad:
                grad.append(param.grad.data.view(-1))
        return torch.cat(grad)


    def fedavg(self, iteration, criterion, gamma, inner_iter=1, record_interval = 2, true_grad=True):
        params = self.server.get_params()
        device = self.server.device
        momentum  = [torch.zeros_like(params["flat_params"]).to(device) for _ in range(len(self.clients))]
        loss = []
        accuracy = []
        grad_norm = []
        for t in range(iteration):

            for i, client in enumerate(self.clients):
                client_params = client.clone_params(params)
                client.set_params(client_params)
                client.train(criterion,num_epoch=inner_iter)
                params_new = client.get_params()
                update_inner = (client_params["flat_params"]-params_new["flat_params"])/(client.optimizer.param_groups[0]['lr'] * inner_iter)

                momentum[i] = self.theta * momentum[i] + (1-self.theta) * update_inner.to(device)
            
            update = sum(momentum) / len(self.clients)
            
            if self.with_gn:
                norm = torch.norm(update,p=2)
                update = update/norm

            params["flat_params"]-= gamma*update
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
                else:
                    grad_norm.append(torch.norm(update,2))
                print(f"{self.algorithm_type} Round {t}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%")
                        
        return loss, accuracy, grad_norm


        

