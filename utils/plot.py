import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

def plot_accuracy_comparison(accuracy_results, title='Test Accuracy vs Iteration Count', 
                           record_interval=2, figsize=(10, 6), use_markers=True):

    plt.figure(figsize=figsize)
    
    for algo_name, accuracies in accuracy_results.items():
        if use_markers:
            plt.plot(np.arange(0, len(accuracies) * record_interval, record_interval)[:len(accuracies)], 
                    accuracies, '-o', label=algo_name, alpha=0.7)
        else:
            plt.plot(np.arange(0, len(accuracies) * record_interval, record_interval)[:len(accuracies)], 
                    accuracies, '-', label=algo_name, alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Iteration Count (T)')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, max(len(acc) for acc in accuracy_results.values()) * record_interval, 
                   max(1, record_interval)))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_loss_comparison(loss_results, title='Training Loss vs Iteration Count', 
                        record_interval=2, figsize=(10, 6), use_log_scale=False, use_markers=True):
 
    plt.figure(figsize=figsize)
    
    for algo_name, losses in loss_results.items():
        if use_markers:
            plt.plot(np.arange(0, len(losses) * record_interval, record_interval)[:len(losses)], 
                    losses, '-o', label=algo_name, alpha=0.7)
        else:
            plt.plot(np.arange(0, len(losses) * record_interval, record_interval)[:len(losses)], 
                    losses, '-', label=algo_name, alpha=0.7)
    
    if use_log_scale:
        plt.yscale('log')
    
    plt.title(title)
    plt.xlabel('Iteration Count (T)')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, max(len(loss) for loss in loss_results.values()) * record_interval, 
                   max(1, record_interval)))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_both_metrics(loss_results, acc_results, title_prefix='', record_interval=2, 
                     figsize=(15, 6), use_log_scale=False):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for algo_name, losses in loss_results.items():
        ax1.plot(np.arange(0, len(losses) * record_interval, record_interval)[:len(losses)], 
                losses, '-o', label=algo_name, alpha=0.7)
    
    if use_log_scale:
        ax1.set_yscale('log')
    
    ax1.set_title(f'{title_prefix}Training Loss')
    ax1.set_xlabel('Iteration Count (T)')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for algo_name, accs in acc_results.items():
        ax2.plot(np.arange(0, len(accs) * record_interval, record_interval)[:len(accs)], 
                accs, '-o', label=algo_name, alpha=0.7)
    
    ax2.set_title(f'{title_prefix}Test Accuracy')
    ax2.set_xlabel('Iteration Count (T)')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_gradient_norm_comparison(grad_norm_results, title='Gradient Norm vs Iteration Count',
                                record_interval=1, figsize=(10, 6), use_log_scale=True):
    plt.figure(figsize=figsize)
    
    for algo_name, grad_norms in grad_norm_results.items():
        plt.plot(np.arange(0, len(grad_norms) * record_interval, record_interval)[:len(grad_norms)], 
                grad_norms, '-o', label=algo_name, alpha=0.7)
    
    if use_log_scale:
        plt.yscale('log')
    
    plt.title(title)
    plt.xlabel('Iteration Count (T)')
    plt.ylabel('Gradient Norm')
    plt.xticks(np.arange(0, max(len(norm) for norm in grad_norm_results.values()) * record_interval, 
                   max(1, record_interval)))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_min_function_values(func_values_results, title='Function Values vs Iteration Count',
                           record_interval=1, figsize=(10, 6), use_log_scale=True):
    plt.figure(figsize=figsize)
    
    for algo_name, func_values in func_values_results.items():
        min_func_values = []
        current_min = float('inf')
        for val in func_values:
            current_min = min(current_min, val)
            min_func_values.append(current_min)
        
        plt.plot(np.arange(0, len(func_values) * record_interval, record_interval)[:len(func_values)], 
                min_func_values, '-o', label=algo_name, alpha=0.7)
    
    if use_log_scale:
        plt.yscale('log')
    
    plt.title(title)
    plt.xlabel('Iteration Count (T)')
    plt.ylabel('Function Value (Min)')
    plt.xticks(np.arange(0, max(len(func) for func in func_values_results.values()) * record_interval, 
                   max(1, record_interval)))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_with_custom_x_axis(y_data_dict, x_values=None, title='Custom Plot', 
                           xlabel='X-axis', ylabel='Y-axis', figsize=(10, 6), use_markers=True):

    plt.figure(figsize=figsize)
    
    if x_values is None:
        for series_name, y_values in y_data_dict.items():
            if use_markers:
                plt.plot(range(len(y_values)), y_values, '-o', label=series_name, alpha=0.7)
            else:
                plt.plot(range(len(y_values)), y_values, '-', label=series_name, alpha=0.7)
    else:
        for series_name, y_values in y_data_dict.items():
            if use_markers:
                plt.plot(x_values[:len(y_values)], y_values, '-o', label=series_name, alpha=0.7)
            else:
                plt.plot(x_values[:len(y_values)], y_values, '-', label=series_name, alpha=0.7)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_simple_accuracy_comparison(accuracy_fedavg, accuracy_normed_fedavg, 
                                  title='Test Accuracy vs Iteration Count', 
                                  max_iteration=40, record_interval=2):

    T = np.arange(0, max_iteration, record_interval)
    
    plt.plot(T[:len(accuracy_fedavg)], accuracy_fedavg, '-o', label='FedAvg', alpha=0.5)
    plt.plot(T[:len(accuracy_normed_fedavg)], accuracy_normed_fedavg, '-o', label='GN FedAvg', alpha=0.5)
    
    plt.xticks(T)
    plt.title(title)
    plt.xlabel('Iteration Count (T)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_data_distribution(dataset_list, num_classes=10, title="Data Distribution Heatmap"):
    num_clients = len(dataset_list)
    
    distribution_matrix = np.zeros((num_clients, num_classes))
    
    for client_idx, dataset in enumerate(dataset_list):
        class_counts = np.zeros(num_classes)
        
        for _, label in dataset:
            class_counts[label.item()] += 1
        
        total_samples = len(dataset)
        if total_samples > 0:
            distribution_matrix[client_idx] = class_counts / total_samples
    
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(
        distribution_matrix,
        annot=True,  
        fmt='.3f',   
        cmap='viridis',  
        xticklabels=[f'Class {i}' for i in range(num_classes)],
        yticklabels=[f'Client {i}' for i in range(num_clients)],
        cbar_kws={'label': 'Data Proportion'},
        vmin=0, vmax=1  
    )
    
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Clients')
    plt.tight_layout()
    plt.show()
    
    return distribution_matrix

def visualize_data_distribution_advanced(dataset_list, num_classes=10, 
                                      title="Data Distribution Heatmap", 
                                      figsize=(12, 8), 
                                      show_values=True):

    num_clients = len(dataset_list)
    
    distribution_matrix = np.zeros((num_clients, num_classes))
    absolute_counts = np.zeros((num_clients, num_classes), dtype=int)
    
    for client_idx, dataset in enumerate(dataset_list):
        class_counts = np.zeros(num_classes, dtype=int)
        
        if hasattr(dataset, 'tensors'):
            _, labels = dataset.tensors
            unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
            for label, count in zip(unique_labels, counts):
                if 0 <= label < num_classes:
                    class_counts[label] += count
        else:
            for i in range(len(dataset)):
                try:
                    item = dataset[i]
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        label = item[1]
                    else:
                        label = item
                    
                    if hasattr(label, 'item'):
                        label_val = label.item()
                    else:
                        label_val = int(label)
                    
                    if 0 <= label_val < num_classes:
                        class_counts[label_val] += 1
                except:
                    continue
        
        absolute_counts[client_idx] = class_counts
        total_samples = len(dataset)
        if total_samples > 0:
            distribution_matrix[client_idx] = class_counts / total_samples
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    sns.heatmap(
        distribution_matrix,
        annot=show_values,
        fmt='.3f',
        cmap='viridis',
        xticklabels=[f'C{i}' for i in range(num_classes)],
        yticklabels=[f'Client {i}' for i in range(num_clients)],
        cbar_kws={'label': 'Proportion'},
        ax=ax1,
        vmin=0, vmax=1
    )
    ax1.set_title('Data Proportion Distribution')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Clients')
    
    sns.heatmap(
        absolute_counts,
        annot=show_values,
        fmt='d',
        cmap='plasma',
        xticklabels=[f'C{i}' for i in range(num_classes)],
        yticklabels=[f'Client {i}' for i in range(num_clients)],
        cbar_kws={'label': 'Sample Count'},
        ax=ax2
    )
    ax2.set_title('Data Count Distribution')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Clients')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return distribution_matrix, absolute_counts

def calculate_heterogeneity_score(distribution_matrix):
    num_clients, num_classes = distribution_matrix.shape
    
    global_dist = np.mean(distribution_matrix, axis=0)
    
    client_divergences = []
    for i in range(num_clients):
        client_dist = distribution_matrix[i]
        client_dist = client_dist + 1e-10
        global_dist = global_dist + 1e-10
        
        kl_div = np.sum(client_dist * np.log(client_dist / global_dist))
        client_divergences.append(kl_div)
    
    heterogeneity_score = np.mean(client_divergences)
    return heterogeneity_score

def visualize_heterogeneity_comparison(dataset_lists_dict, num_classes=10):
    methods = list(dataset_lists_dict.keys())
    scores = []
    matrices = {}
    
    for method, dataset_list in dataset_lists_dict.items():
        num_clients = len(dataset_list)
        distribution_matrix = np.zeros((num_clients, num_classes))
        
        for client_idx, dataset in enumerate(dataset_list):
            class_counts = np.zeros(num_classes)
            for _, label in dataset:
                class_counts[label.item()] += 1
            
            total_samples = len(dataset)
            if total_samples > 0:
                distribution_matrix[client_idx] = class_counts / total_samples
        
        score = calculate_heterogeneity_score(distribution_matrix)
        scores.append(score)
        matrices[method] = distribution_matrix
    
    fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 5))
    if len(methods) == 1:
        axes = [axes]
    
    for i, method in enumerate(methods):
        sns.heatmap(
            matrices[method],
            annot=True,
            fmt='.2f',
            cmap='viridis',
            xticklabels=[f'C{i}' for i in range(num_classes)],
            yticklabels=[f'Client {i}' for i in range(len(dataset_lists_dict[method]))],
            cbar_kws={'label': 'Proportion'},
            ax=axes[i],
            vmin=0, vmax=1
        )
        axes[i].set_title(f'{method}\nHeterogeneity: {scores[i]:.3f}')
        axes[i].set_xlabel('Classes')
        axes[i].set_ylabel('Clients')
    
    plt.tight_layout()
    plt.show()
    
    return scores, matrices

def visualize_client_distributions(client_distributions, num_classes=10, 
                                  title="Client Distributions Heatmap", 
                                  figsize=(10, 6)):

    num_clients = client_distributions.shape[0]
    
    if torch.is_tensor(client_distributions):
        distributions_np = client_distributions.numpy()
    else:
        distributions_np = client_distributions
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        distributions_np,
        annot=False,  
        fmt='.3f',   
        cmap='viridis',  
        xticklabels=[f'Class {i}' for i in range(num_classes)],
        yticklabels=[f'Client {i}' for i in range(num_clients)],
        cbar_kws={'label': 'Probability'},
        vmin=0, vmax=1  
    )
    
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Clients')
    plt.tight_layout()
    plt.show()
    
    return distributions_np
