import sys, os
import argparse
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("../..")
from src.datasets import StitchGridDataset
from src.resnet import resnetseg34, train_rnseg

def aba_data(img_train, lbl_train, img_val, lbl_val):

	patch_size = 256
	xform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(patch_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	stitch_train = StitchGridDataset(img_train, lbl_train, xform)
	stitch_val = StitchGridDataset(img_val, lbl_val, xform)

	class_names = ["Midbrain","Isocortex","Medulla","Striatum",
		"C. nuclei","C. cortex","Thalamus","Olf. areas",
		"Cort. sub.","Pons","Pallidum","Hipp. form.","Hypothal."]

	return stitch_train, stitch_val, class_names, 32, 49, patch_size

def maniatis_data(img_train, lbl_train, img_val, lbl_val):
	
	patch_size = 256
	xform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(patch_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	stitch_train = StitchGridDataset(img_train, lbl_train, xform)
	stitch_val = StitchGridDataset(img_val, lbl_val, xform)

	class_names = ["Vent. Med. White", "Vent. Horn", "Vent. Lat. White", "Med. Gray", 
		"Dors. Horn", "Dors. Edge", "Med. Lat. White", "Vent. Edge", "Dors. Med. White",
		"Cent. Canal", "Lat. Edge"]

	return stitch_train, stitch_val, class_names, 35, 33, patch_size


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help='"aba" or "maniatis"')
	parser.add_argument('imgtrain', type=str, help='Path to training set image directory')
	parser.add_argument('lbltrain', type=str, help='Path to training set label directory')
	parser.add_argument('imgval', type=str, help='Path to validation set image directory')
	parser.add_argument('lblval', type=str, help='Path to validation set label directory')
	parser.add_argument('-o', '--outfile', required=True, help='Path to save model output')
	parser.add_argument("-l", "--lr", type=float, default=None, help="Learning rate")
	args = parser.parse_args()

	if args.dataset == "aba":
		train, val, class_names, h_st, w_st, patch_size = aba_data(
			args.imgtrain, args.lbltrain, args.imgval, args.lblval)
	else:
		train, val, class_names, h_st, w_st, patch_size = maniatis_data(
			args.imgtrain, args.lbltrain, args.imgval, args.lblval)		
	print("Training set: %d arrays" % len(train))
	print("Validation set: %d arrays" % len(val))

	# Data Loaders
	batch_size = 1
	data_loaders = {
		"train": DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True),
		"val": DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=True)
	}

	# Model Instantiation
	n_class = len(class_names)
	rns34 = resnetseg34(n_class, (h_st, w_st), thin=4)

	# Learning rate
	if args.lr is None:
		args.lr = 10 ** (np.random.uniform(-4,-3))
	print("Learning Rate: %.4g" % args.lr)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(rns34.parameters(), lr=args.lr)

	# Perform fitting and save model
	train_rnseg(rns34, data_loaders, criterion, optimizer, 200,
		outfile=args.outfile)
