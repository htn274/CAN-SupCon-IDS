import torch
import torch.nn as nn
import torch.nn.functional as F

class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding='same'), # 29x29x32
            nn.Conv2d(32, 32, 3), # 27x27x32
            nn.MaxPool2d(2, stride=2), #13x13x32
            nn.Conv2d(32, 64, 1), #13x13x64
            nn.Conv2d(64, 128, 3, padding='same'), # 13x13x128
            nn.Conv2d(128, 128, 3, padding='same') # 13x13x128
        )
    def forward(self, x):
        return self.conv(x)

class InceptionresenetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Conv2d(128, 32, 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32, 32, 3, padding='same'),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.Conv2d(32, 32, 3, padding='same')
        )
        self.linear = nn.Conv2d(96, 128, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = [
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ]
        #print([e.shape for e in residual])
        residual = torch.cat(residual, 1)
        residual = self.linear(residual)
        output = self.relu(x + residual)
        return output
    
class ReductionA(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.MaxPool2d(3, stride=2)
        self.branch2 = nn.Conv2d(128, 192, 3, stride=2)
        self.branch3 = nn.Sequential(
                nn.Conv2d(128, 96, 1),
                nn.Conv2d(96, 96, 3, padding='same'),
                nn.Conv2d(96, 128, 3, stride=2)
        )
        
    def forward(self, x):
        x = [
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ]
        return torch.cat(x, 1)

class InceptionresnetB(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Conv2d(448, 64, 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(448, 64, 1),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding='same')
        )
        self.linear = nn.Conv2d(64*2, 448, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = [
            self.branch1(x),
            self.branch2(x)
        ]
        residual = torch.cat(residual, 1)
        residual = self.linear(residual)
        output = self.relu(x + residual)
        return output
    
class ReductionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.MaxPool2d(3)
        self.branch2 = nn.Sequential(
                nn.Conv2d(448, 128, 1),
                nn.Conv2d(128, 192, 3, stride=2)
            )
        self.branch3 = nn.Sequential(
                nn.Conv2d(448, 128, 1),
                nn.Conv2d(128, 128, 3, stride=2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(448, 128, 1),
            nn.Conv2d(128, 128, 3),
            nn.Conv2d(128, 128, 3)
        )
        
    def forward(self, x):
        x = [
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
        ]
        # print([e.shape for e in x])
        return torch.cat(x, 1)

class InceptionResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem()
        self.inceptionresA = InceptionresenetA()
        self.reductionA = ReductionA()
        self.inceptionresB = InceptionresnetB()
        self.reductionB = ReductionB()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(1 - 0.8)
        # self.linear = nn.Linear(896, n_classes)
        
    def forward(self, x):
        x = self.stem(x)
        # print('Stem: ', x.shape)
        x = self.inceptionresA(x)
        # print('Incetion Res A: ', x.shape)
        x = self.reductionA(x)
        # print('Reduction A: ', x.shape)
        x = self.inceptionresB(x)
        # print('Inception Res B: ', x.shape)
        x = self.reductionB(x)
        # print('Reduction B: ', x.shape)
        x = self.avgpool(x)
        # print('Avg pool: ', x.shape)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        # print('Final: ', x.shape)
        return x

class SupIncepResnet(nn.Module):
    def __init__(self, num_classes):
        super(SupIncepResnet, self).__init__()
        self.encoder = InceptionResnet()
        self.fc = nn.Linear(896, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))