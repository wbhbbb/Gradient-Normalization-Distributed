
import torch
import torch.distributions as dist
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

def create_doubly_stochastic_matrix(num_clients, num_classes, alpha, max_iter=100, tol=1e-6):
    """
    To generate a doubly stochastic matrix using the Sinkhorn-Knopp algorithm
    """
    alpha_vec = torch.ones(num_classes) / num_classes * alpha
    dirichlet = dist.Dirichlet(alpha_vec)
    
    matrix = torch.stack([dirichlet.sample() for _ in range(num_clients)])
    
    matrix = matrix + 1e-8
    
    for _ in range(max_iter):
        row_sums = matrix.sum(dim=1, keepdim=True)
        matrix = matrix / row_sums
        
        col_sums = matrix.sum(dim=0, keepdim=True)
        matrix = matrix / col_sums
        
        row_check = torch.max(torch.abs(row_sums - 1))
        col_check = torch.max(torch.abs(col_sums - 1))
        if row_check < tol and col_check < tol:
            break
    
    return matrix

def split_doubly_stochastic_direct(train_dataset, alpha, num_clients, num_classes=10):
    train_size = len(train_dataset)
    subsets = [[] for _ in range(num_classes)]
    final_dataset = [[] for _ in range(num_clients)]
    
    for i in range(train_size):
        image, label = train_dataset[i]
        subsets[label].append((image, label))
    
    doubly_stochastic = create_doubly_stochastic_matrix(
        num_clients, num_classes, alpha
    )
    
    total_per_client = train_size // num_clients
    
    for class_idx in range(num_classes):
        class_size = len(subsets[class_idx])
        class_allocation = (doubly_stochastic[:, class_idx] * class_size).long()
        
        total = class_allocation.sum().item()
        if total < class_size:
            diff = class_size - total
            remainders = (doubly_stochastic[:, class_idx] * class_size) - class_allocation.float()
            _, indices = torch.topk(remainders, diff)
            for idx in indices:
                class_allocation[idx] += 1
        
        permutation = torch.randperm(class_size)
        start_idx = 0
        
        for client_idx in range(num_clients):
            num_samples = class_allocation[client_idx].item()
            if num_samples > 0:
                end_idx = start_idx + num_samples
                client_indices = permutation[start_idx:end_idx]
                for idx in client_indices:
                    final_dataset[client_idx].append(subsets[class_idx][idx])
                start_idx = end_idx

    datasets_tensor = []
    for i in range(num_clients):
        images = torch.stack([x[0] for x in final_dataset[i]])
        labels = torch.tensor([x[1] for x in final_dataset[i]], dtype=torch.int64)
        datasets_tensor.append(TensorDataset(images, labels))
    
    return datasets_tensor, doubly_stochastic


def split_dirichlet(train_dataset, alpha, num_clients, num_classes=10):
    """
    Using a Dirichlet distribution to perform Non-IID data partitioning
    """
    train_size = len(train_dataset)
    subsets = [[] for _ in range(num_classes)]
    
    for i in range(train_size):
        image, label = train_dataset[i]
        subsets[label].append((image, label))
    
    dirichlet = dist.Dirichlet(torch.ones(num_classes) * alpha)
    client_distributions = torch.stack([dirichlet.sample() for _ in range(num_clients)])
    final_dataset = [[] for _ in range(num_clients)]
    
    for client_idx in range(num_clients):
        client_dist = client_distributions[client_idx]
        
        for class_idx in range(num_classes):
            class_size = len(subsets[class_idx])
            num_samples = int(client_dist[class_idx] * class_size)
            
            num_samples = min(num_samples, len(subsets[class_idx]))
            
            indices = torch.randperm(len(subsets[class_idx]))[:num_samples]
            for idx in indices:
                final_dataset[client_idx].append(subsets[class_idx][idx.item()])
    
    datasets_tensor = []
    for i in range(num_clients):
        if len(final_dataset[i]) > 0: 
            images = torch.stack([x[0] for x in final_dataset[i]])
            labels = torch.tensor([x[1] for x in final_dataset[i]], dtype=torch.int64)
            datasets_tensor.append(TensorDataset(images, labels))
        else:
            datasets_tensor.append(TensorDataset(torch.empty(0, 28, 28), torch.empty(0, dtype=torch.int64)))
    
    return datasets_tensor, client_distributions