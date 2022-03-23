import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
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
        
        self.head = nn.Sequential(
                    nn.Linear(64*3*3, 256), nn.ReLU(),
                    nn.Linear(256, feat_dim))
        
    def encoder(self, x):
        feat = self.convnet(x)
        feat = torch.flatten(feat, 1)
        return feat
        
    def forward(self, x):
        feat = self.encoder(x)
        feat = self.head(feat)
        feat = F.normalize(feat, dim=1)
        return feat
    
class LinearClassifier(nn.Module):
    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(feat_dim, n_classes)
        
    def forward(self, x):
        output = self.fc(x)
        return output
    
class BaselineCNNClassifier(nn.Module):
    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_classes = n_classes
        self.encoder = CNNEncoder(feat_dim)
        self.fc = nn.Linear(feat_dim, n_classes)
        
    def forward(self, x):
        feat = self.encoder(x)
        out = self.fc(feat)
        return out
        
class BaselineCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.convnet = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding='same'), nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding='same'),nn.ReLU(),
                    nn.BatchNorm2d(16),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(16, 32, 3, padding='same'),nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding='same'),nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.MaxPool2d(2, stride=2),
                    nn.Dropout2d(1 - 0.8)
                )
        self.fc = nn.Sequential(
                    nn.Linear(32*7*7, 512), nn.ReLU(),
                    nn.Dropout2d(1 - 0.8),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Dropout2d(1 - 0.8),
                    nn.Linear(256, self.n_classes),
                )
        
    def forward(self, x):
        output = x
        output = self.convnet(output)
        output = output.view(output.size()[0], -1) 
        output = self.fc(output)
        return output