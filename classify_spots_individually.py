import h5py
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sklearn
import time, copy

hf = h5py.File('/mnt/home/thamamsy/data/mouse_spinal/data.h5', 'r')
X_train, X_val, y_train, y_val = train_test_split(hf.get('genes').value, hf.get('annotation').value, test_size=0.2)
#X_train = sklearn.preprocessing.normalize(X_train, norm='l2', axis=1)
#X_val = sklearn.preprocessing.normalize(X_val, norm='l2', axis=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = torch.from_numpy(X_train).to(device)
X_val = torch.from_numpy(X_val).to(device)
y_train = torch.from_numpy(y_train).to(device)
y_val = torch.from_numpy(y_val).to(device)

dtrain = torch.utils.data.TensorDataset(X_train, y_train)
dtest = torch.utils.data.TensorDataset(X_val, y_val)
loadtrain = DataLoader(dtrain, batch_size = 200)
loadtest = DataLoader(dtest, batch_size = 200)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(23860, 300)
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50, 11)


    def forward(self, x):
        x = x.view(-1, 23860)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=.2)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


clf = Net().to(device)
#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
optimizer = optim.Adam(clf.parameters(), lr = 0.001)

for epoch in range(200):
    losses = []
    # Train
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(loadtrain):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()                        # Zero the gradients
        outputs = clf(inputs.float())                # Forward pass
        loss = criterion(outputs, targets.long())    # Compute the Loss
        loss.backward()                              # Compute the Gradients
        optimizer.step()                             # Updated the weights
        losses.append(loss.item())
        end = time.time()

        #if batch_idx % 100 == 0:
        #    print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (batch_idx, np.mean(losses), end - start))

    start = time.time()
    # Evaluate
    clf.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loadtest):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = clf(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        print('Epoch : %d Test Acc : %.3f' % (epoch, 100.*correct/total))
        print('--------------------------------------------------------------')
    clf.train()
