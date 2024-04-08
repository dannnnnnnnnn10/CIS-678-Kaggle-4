import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

train_adt = np.genfromtxt("train_ADT.csv", delimiter=",", dtype=float, skip_header=1)
train_rna = np.genfromtxt("train_RNA.csv", delimiter=",", dtype=float, skip_header=1)

train_adt = train_adt.T
train_rna = train_rna.T

train_adt = np.log1p(train_adt)
train_rna = np.log1p(train_rna)

train_adt = torch.from_numpy(train_adt)
train_rna = torch.from_numpy(train_rna)

train_adt = train_adt.to(torch.float32)
train_rna = train_rna.to(torch.float32)

train_rna = train_rna.to_sparse()

adt_ds = TensorDataset(train_adt)
rna_ds = TensorDataset(train_rna)

adt_dl = DataLoader(adt_ds, batch_size=64)
rna_dl = DataLoader(rna_ds, batch_size=64)

print("Created dataloaders with no issues")
print(adt_dl)
print(rna_dl)

torch.manual_seed(12345)

device = "cpu"

class rna_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(10000, 256),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits
    
class rna_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Linear(256, 10000),
            nn.ReLU()
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

class adt_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(25, 20),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Linear(20, 16),
            nn.ReLU()
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

class adt_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(16, 20),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Linear(20, 25),
            nn.ReLU()
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

class variational_auto_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(16, 10),
            nn.ReLU(),
            nn.BatchNorm1d(),
            nn.Linear(10, 16),
            nn.ReLU()
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

rna_enc = rna_encoder().to(device)
rna_dec = rna_decoder().to(device)
adt_enc = adt_encoder().to(device)
adt_dec = adt_decoder().to(device)
vac = variational_auto_encoder().to(device)

loss_fn = nn.MSELoss()
rna_enc_optim = optim.Adam(rna_enc.parameters(), lr = 0.001)
rna_dec_optim = optim.Adam(rna_dec.parameters(), lr = 0.001)
adt_enc_optim = optim.Adam(adt_enc.parameters(), lr = 0.001)
adt_dec_optim = optim.Adam(adt_dec.parameters(), lr = 0.001)
vac_optim = optim.Adam(vac.parameters(), lr=0.001)

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