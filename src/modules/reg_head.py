import torch.nn as nn
import torch

class LlamaRegressionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)        
        x = self.sigmoid(x)      
        x = x * 6 + 1
        return x
