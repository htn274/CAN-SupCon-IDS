import numpy as np
import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
 
class CANDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = Path(root_dir) / ('train' if is_train else 'val')
        self.is_train = is_train
        self.transform = transform
        self.total_size = len(os.listdir(self.root_dir))
            
    def __getitem__(self, idx):
        filename = f'{idx}.npz'
        filename = self.root_dir / filename
        data = np.load(filename)
        X, y = data['X'], data['y']
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return X_tensor, y_tensor

    def __len__(self):
        return self.total_size
