import sys, os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("../../")
from src.datasets import PatchDataset
from src.patch_classifier import train_model


parser = argparse.ArgumentParser()
parser.add_argument('imgtrain', type=str, help="Path to training image directory")
parser.add_argument('lbltrain', type=str, help="Path to training label directory")
parser.add_argument('imgval', type=str, help="Path to validation image directory")
parser.add_argument('lblval', type=str, help="Path to validation label directory")
parser.add_argument("-p", "--pretrain", action="store_true")
args = parser.parse_args()

if args.pretrain:
	model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
else:
	model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)

xform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(patch_size),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set = PatchDataset(args.imgtrain, args.lbltrain, xform)
val_set = PatchDataset(args.imgval, args.lblval, xform)

dataloaders = {
	'train': DataLoader(train_set, batch_size=32, shuffle=True),
	'val': DataLoader(val_set, batch_size=32, shuffle=True)
}


lr = 0.001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


best_model, train_hist = train_model(model, dataloaders, loss, optimizer, 50)