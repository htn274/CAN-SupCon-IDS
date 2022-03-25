import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    output_shape = 64*3*3
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding='same'), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding='same'),nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(16, 32, 3, padding='same'),nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding='same'),nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, 3, padding='same'),nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding='same'),nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, stride=2))
       
    def forward(self, x):
        feat = self.convnet(x)
        feat = torch.flatten(feat, 1)
        return feat
    
class SupConCNN(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.encoder = CNNEncoder()
        self.feat_dim = feat_dim
        dim_in = self.encoder.output_shape
        self.head = nn.Sequential(
                    nn.Linear(dim_in, 256), nn.ReLU(),
                    nn.Linear(256, feat_dim))
       
    def forward(self, x):
        feat = self.encoder(x)
        feat = self.head(feat)
        feat = F.normalize(feat, dim=1)
        return feat
    
class LinearClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        feat_dim = CNNEncoder.output_shape
        self.n_classes = n_classes
        self.fc = nn.Linear(feat_dim, n_classes)
        
    def forward(self, x):
        output = self.fc(x)
        return output
    
class BaselineCNNClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = CNNEncoder()
        feat_dim = CNNEncoder.output_shape
        self.n_classes = n_classes
        self.fc = nn.Linear(feat_dim, n_classes)
        
    def forward(self, x):
        feat = self.encoder(x)
        out = self.fc(feat)
        return out
        