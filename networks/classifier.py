import torch
import torch.nn as nn
import torch.nn.functional as F
    
class LinearClassifier(nn.Module):
    def __init__(self, n_classes, feat_dim, init=False):
        super().__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(feat_dim, n_classes)
        if init:
            torch.nn.init.xavier_normal_(self.fc.weight)
            torch.nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x):
        output = self.fc(x)
        return output