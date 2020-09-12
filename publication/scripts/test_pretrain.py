import sys, os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("../..")
from datasets import PatchDataset
from patch_classifier import train_model


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pretrain", action="store_true")
args = parser.parse_args()

if args.pretrain:
	model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
	#outfile = "../data/resnet18_pretrain_aba128.pth"
else:
	model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
	#outfile = "../data/resnet18_rinit_aba128.pth"


data_dir = "/mnt/ceph/users/adaly/datasets/aba_stdataset_20200212/"
img_train = os.path.join(data_dir, "imgs128_train")
lbl_train = os.path.join(data_dir, "lbls128_train")
train_set = PatchDataset(img_train, lbl_train)

img_val = os.path.join(data_dir, "imgs128_val")
lbl_val = os.path.join(data_dir, "lbls128_val")
val_set = PatchDataset(img_val, lbl_val)

dataloaders = {
	'train': DataLoader(train_set, batch_size=32, shuffle=True),
	'val': DataLoader(val_set, batch_size=32, shuffle=True)
}


lr = 0.001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


#best_model, train_hist = train_model(model, dataloaders, loss, optimizer, 50,
#	outfile=outfile)
best_model, train_hist = train_model(model, dataloaders, loss, optimizer, 50)