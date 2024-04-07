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
            nn.Linear(10000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits
    
class rna_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10000)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

class adt_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(25, 15),
            nn.ReLU(),
            nn.Linear(15, 10)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

class adt_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(10, 15),
            nn.ReLU(),
            nn.Linear(15, 25)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

rna_enc = rna_encoder().to(device)
rna_dec = rna_decoder().to(device)
adt_enc = adt_encoder().to(device)
adt_dec = adt_decoder().to(device)

loss_fn = nn.MSELoss()
rna_enc_optim = optim.Adam(rna_enc.parameters(), lr = 0.001)
rna_dec_optim = optim.Adam(rna_dec.parameters(), lr = 0.001)
adt_enc_optim = optim.Adam(adt_enc.parameters(), lr = 0.001)
adt_dec_optim = optim.Adam(adt_dec.parameters(), lr = 0.001)

def train(dataloader, enc, dec, loss_fn, enc_optimizer, dec_optimizer):
    size = len(dataloader.dataset)
    enc.train()
    dec.train()
    for batch, (X) in enumerate(dataloader):
        pred = rna_enc(X)
        pred = rna_dec(pred)

        loss = loss_fn(pred, X)

        loss.backward()
        dec_optimizer.step()
        dec_optimizer.zero_grad()
        enc_optimizer.step()
        enc_optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")