
import torch
import torch.distributions as dist

class AsynSGD:

    SUPPORTED_ALGORITHMS = ["AsySGD", "AsySGD-GN-M"]
    def __init__(self,clients,server,theta,algorithm_type):
        self.clients = clients
        self.server = server
        self.algorithm_type = algorithm_type

        if algorithm_type not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm_type}. Supported: {self.SUPPORTED_ALGORITHMS}")
        
        if algorithm_type=="AsySGD":
            self.theta=0.0
            self.with_gn = False
        else:
            self.theta=theta 
            self.with_gn = True




    def asysgd(self,iteration, criterion, gamma, inner_iter=1, record_interval = 2):

        params = self.server.get_params()
        device = self.server.device
        num_clients=len(self.clients)
        momentum  = [torch.zeros_like(params["flat_params"]).to(device) for _ in range(num_clients)]
        loss = []
        accuracy = []
        grad_norm = []
        

        concentration = torch.tensor([1.0])
        rate = torch.tensor([2.0])
        gamma_dict = dist.Gamma(concentration, rate)
        tau_mean = gamma_dict.sample((num_clients,))
        # print(tau_mean)
        tau = torch.zeros((num_clients,3))
        for i in range(num_clients):
            tau[i,2] = torch.max(torch.tensor([1,tau_mean[i].item()+0.1*torch.randn(1).item()]))
        tau[:,0] = -1
        # print(tau)
        global_params = [self.server.clone_params(params) for _ in range(iteration+1)]

        for t in range(iteration):
        
            for i in range(num_clients):
                if t>=tau[i][2]:
                    tau[i][0] = tau[i][1]
                    tau[i][1] = tau[i][2]
                    a = tau[i][2]
                    b = int(a.item())+1
                    c = a + max([0,tau_mean[i].item()+0.1*torch.randn(1).item()]) 
                    if c<b:
                        tau[i][2] = b
                    else:
                        tau[i][2] = c

            for i, client in enumerate(self.clients):
                tau1 = int(tau[i][0].item())+1
                client_params = global_params[tau1]
                client.set_params(client_params)
                client.train(criterion,num_epoch=inner_iter)
                params_new = client.get_params()
                update_inner = (client_params["flat_params"]-params_new["flat_params"])/(client.optimizer.param_groups[0]['lr'] * inner_iter)
                momentum[i] = self.theta * momentum[i] + (1-self.theta) * update_inner.to(device)
    
        
            update = sum(momentum) / num_clients
            
            if self.with_gn:
                norm = torch.norm(update,p=2)
                update = update/norm

            params["flat_params"]-= gamma*update
            self.server.set_params(params)
            global_params[t+1]=self.server.clone_params(params)


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

        return loss, accuracy