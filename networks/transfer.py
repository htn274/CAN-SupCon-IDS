import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class TransferModel(nn.Module):
    def __init__(self, feat_extractor, classifier):
        super().__init__()
        self.encoder = copy.deepcopy(feat_extractor)
        self.classifier = copy.deepcopy(classifier)
        
    def forward(self, x, return_feat=False):
        feat = self.encoder(x)
        output = self.classifier(feat)
        if return_feat:
            return feat, output
        return output