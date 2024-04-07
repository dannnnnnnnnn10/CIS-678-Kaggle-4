import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(12345)

device = "cpu"

class rna_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(784, 160),
            nn.ReLU(),
            nn.Linear(160, 45),
            nn.ReLU(),
            nn.Linear(45, 10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits
    
class rna_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(784, 160),
            nn.ReLU(),
            nn.Linear(160, 45),
            nn.ReLU(),
            nn.Linear(45, 10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

class adt_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(784, 160),
            nn.ReLU(),
            nn.Linear(160, 45),
            nn.ReLU(),
            nn.Linear(45, 10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

class adt_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(784, 160),
            nn.ReLU(),
            nn.Linear(160, 45),
            nn.ReLU(),
            nn.Linear(45, 10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

rna_enc = rna_encoder().to(device)
