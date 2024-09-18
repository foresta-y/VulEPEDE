from typing import Any
import torch
import numpy as np
from torch.utils.data import Dataset
import os

def pad_or_truncate(feature, max_len):
    if len(feature) > max_len:
        return feature[:max_len]
    elif len(feature) < max_len:
        padding = np.zeros((max_len - len(feature), feature.shape[1]))
        return np.vstack((feature, padding))
    else:
        return feature

class CustomDataset(Dataset):
    def __init__(self, filename, pdg, cddf, labels, pdg_max_len, cddf_max_len, cddf_dim):
        self.filename = filename
        self.pdg = pdg
        self.cddf = cddf
        self.labels = labels
        self.pdg_max_len = pdg_max_len
        self.cddf_max_len = cddf_max_len
        self.cddf_dim = cddf_dim
    
    def __getitem__(self, idx):
        # assert len(self.cddf[idx]) == 2, "CDDE data must contain two features (CDG and DDG)"
        pdg_feature = pad_or_truncate(np.array(self.pdg[idx]), self.pdg_max_len)
        # ddg_feature = pad_or_truncate(np.array(self.cddf[idx][0]), self.cddf_max_len)
        if len(self.cddf[idx][0]) == 0:
            ddg_feature = np.zeros((self.cddf_max_len, self.cddf_dim * 2))
        else:
            ddg_feature = pad_or_truncate(np.array(self.cddf[idx][0]), self.cddf_max_len)
        
        if len(self.cddf[idx][1]) == 0:
            cdg_feature = np.zeros((self.cddf_max_len, ddg_feature.shape[1]))
        else:
            cdg_feature = pad_or_truncate(np.array(self.cddf[idx][1]), self.cddf_max_len)
        
        pdg_feature = torch.tensor(np.array(pdg_feature), dtype=torch.float32)
        
        cddf_feature = torch.tensor(np.stack((cdg_feature, ddg_feature), axis=0), dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return self.filename[idx], pdg_feature, cddf_feature, label
    
    def __len__(self) -> int:
        return len(self.labels)
    
    