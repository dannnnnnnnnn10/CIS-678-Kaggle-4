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
test_rna = np.genfromtxt("test_RNA.csv", delimiter=",", dtype=float, skip_header=1)

train_adt = train_adt.T
train_rna = train_rna.T
test_rna = test_rna.T

train_adt = np.log1p(train_adt)
train_rna = np.log1p(train_rna)
test_rna = np.log1p(test_rna)

train_adt = torch.from_numpy(train_adt)
train_rna = torch.from_numpy(train_rna)
test_rna = torch.from_numpy(test_rna)

train_adt = train_adt.to(torch.float32)
train_rna = train_rna.to(torch.float32)
test_rna = test_rna.to(torch.float32)

adt_ds = TensorDataset(train_adt)
rna_ds = TensorDataset(train_rna)

adt_dl = DataLoader(adt_ds, batch_size=512)
rna_dl = DataLoader(rna_ds, batch_size=512)

print("Created dataloaders with no issues")

torch.manual_seed(12345)

device = "cpu"

class rna_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(10000, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
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
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
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
            nn.BatchNorm1d(20),
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
            nn.BatchNorm1d(20),
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
            nn.BatchNorm1d(10),
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

def train(rna_dl, adt_dl, rna_enc, rna_dec, adt_enc, adt_dec, vac, loss_fn, rna_enc_optimizer, rna_dec_optimizer, adt_enc_optimizer, adt_dec_optimizer, vac_optimizer):

    rna_enc.train()
    rna_dec.train()
    adt_enc.train()
    adt_dec.train()
    vac.train()

    for item1, item2 in zip(enumerate(rna_dl), enumerate(adt_dl)):
        batch1, X1 = item1
        batch2, X2 = item2
        pred = rna_enc(X1)
        pred = vac(pred)
        pred = rna_dec(pred)

        loss = loss_fn(pred, X1)

        loss.backward()
        rna_dec_optimizer.step()
        rna_dec_optimizer.zero_grad()
        vac_optimizer.step()
        vac_optimizer.zero_grad()
        rna_enc_optimizer.step()
        rna_enc_optimizer.zero_grad()

        pred = adt_enc(X2)
        pred = vac(pred)
        pred = adt_dec(pred)

        loss = loss_fn(pred, X2)

        loss.backward()
        adt_dec_optimizer.step()
        adt_dec_optimizer.zero_grad()
        vac_optimizer.step()
        vac_optimizer.zero_grad()
        adt_enc_optimizer.step()
        adt_enc_optimizer.zero_grad()

        print("Epoch " + str(num) + " done!")

def test(dataloader, rna_enc, adt_dec, vac):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    rna_enc.eval()
    adt_dec.eval()
    vac.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = rna_enc(X)
            pred = vac(pred)
            pred = adt_dec(pred)
            diff = nn.BCELoss()
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(str(100*correct))

epochs = 10
for t in range(epochs):
    train(rna_dl, adt_dl, rna_enc, rna_dec, adt_enc, adt_dec, vac, loss_fn, rna_enc_optim, rna_dec_optim, adt_enc_optim, adt_dec_optim, vac_optim)

with torch.no_grad():
    pred = rna_enc(test_rna)
    pred = vac(pred)
    pred = adt_dec(pred)
    pred = pred.detach().numpy()

results = pred.reshape((-1, 1))

with open("output.csv", 'w', newline='') as csvfile:
    outputwriter = csv.writer(csvfile, quotechar='"', delimiter=",")
    outputwriter.writerow(['Id'] + ['Expected'])
    for n in range(results.size):
        outputwriter.writerow(['ID_' + str(int(n+1))] + [str(float(results[n][0]))])