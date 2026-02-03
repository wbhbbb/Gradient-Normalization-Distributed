import torch
import torch.nn as nn

class LogisticRegressionBinary(nn.Module):
    
    def __init__(self, input_size):
        super(LogisticRegressionBinary, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        
        logits = self.linear(x)
        
        return logits.squeeze(1)  
