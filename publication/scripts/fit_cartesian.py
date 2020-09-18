import sys, os
import argparse
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("../..")
from src.datasets import PatchDataset, PatchGridDataset
from src.gridnet_patches import GridNet
from src.patch_classifier import densenet121, densenet_preprocess
from src.training import train_gnet_finetune
from src.densenet import DenseNet


def aba_data(img_train, lbl_train, img_val, lbl_val):
	
	patch_size = 256
	xform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(patch_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	atonce_patch_limit = 32

	patch_train = PatchDataset(img_train, lbl_train, xform)
	grid_train = PatchGridDataset(img_train, lbl_train, xform)

	patch_val = PatchDataset(img_val, lbl_val, xform)
	grid_val = PatchGridDataset(img_val, lbl_val, xform)

	class_names = ["Midbrain","Isocortex","Medulla","Striatum",
		"C. nuclei","C. cortex","Thalamus","Olf. areas",
		"Cort. sub.","Pons","Pallidum","Hipp. form.","Hypothal."]

	return patch_train, patch_val, grid_train, grid_val, class_names, 32, 49, patch_size, atonce_patch_limit


def maniatis_data(img_train, lbl_train, img_val, lbl_val):

	patch_size = 256
	xform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(patch_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	atonce_patch_limit = 32

	patch_train = PatchDataset(img_train, lbl_train, xform)
	grid_train = PatchGridDataset(img_train, lbl_train, xform)

	patch_val = PatchDataset(img_val, lbl_val, xform)
	grid_val = PatchGridDataset(img_val, lbl_val, xform)

	class_names = ["Vent. Med. White", "Vent. Horn", "Vent. Lat. White", "Med. Gray", 
		"Dors. Horn", "Dors. Edge", "Med. Lat. White", "Vent. Edge", "Dors. Med. White",
		"Cent. Canal", "Lat. Edge"]

	return patch_train, patch_val, grid_train, grid_val, class_names, 35, 33, patch_size, atonce_patch_limit

# Tracking down SEGFAULT
import torch
import torch.nn as nn
import src.gridnet_patches as gn

import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help='"aba" or "maniatis"')
	parser.add_argument('imgtrain', type=str, help='Path to training set image directory')
	parser.add_argument('lbltrain', type=str, help='Path to training set label directory')
	parser.add_argument('imgval', type=str, help='Path to validation set image directory')
	parser.add_argument('lblval', type=str, help='Path to validation set label directory')
	parser.add_argument('-o', '--outfile', required=True, help='Path to save model output')
	parser.add_argument("-l", "--lr", type=float, default=None, help="Learning rate")
	parser.add_argument("-a", "--alpha", type=float, default=None, help="Alpha")
	parser.add_argument("-i", "--index", type=int, default=None, help="Index in job array (for repeated fittings)")
	args = parser.parse_args()

	if args.dataset == "aba":
		patch_train, patch_val, grid_train, grid_val, class_names, h_st, w_st, patch_size, atonce_patch_limit = aba_data(
			args.imgtrain, args.lbltrain, args.imgval, args.lblval)
	else:
		patch_train, patch_val, grid_train, grid_val, class_names, h_st, w_st, patch_size, atonce_patch_limit = maniatis_data(
			args.imgtrain, args.lbltrain, args.imgval, args.lblval)		
	print("Training set: %d arrays (%d patches)" % (len(grid_train), len(patch_train)))
	print("Validation set: %d arrays (%d patches)" % (len(grid_val), len(patch_val)))

	# Data Loaders
	batch_size = 1
	patch_loaders = {
		"train": DataLoader(patch_train, batch_size=32, shuffle=True, pin_memory=True),
		"val": DataLoader(patch_val, batch_size=32, shuffle=True, pin_memory=True)
	}
	grid_loaders = {
		"train": DataLoader(grid_train, batch_size=batch_size, shuffle=True, pin_memory=True),
		"val": DataLoader(grid_val, batch_size=batch_size, shuffle=True, pin_memory=True)
	}

	# Model Instantiation
	n_class = len(class_names)
	f = DenseNet(num_classes=n_class, small_inputs=False, efficient=False,
		growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0)
	g = GridNet(f, patch_shape=(3,patch_size,patch_size), grid_shape=(h_st, w_st), n_classes=n_class, 
		use_bn=False, atonce_patch_limit=atonce_patch_limit)

	# Learning rate
	if args.lr is None:
		args.lr = 10 ** (np.random.uniform(-4,-3))
	if args.alpha is None:
		args.alpha = np.random.random() * 0.1
	print("Learning Rate: %.4g" % args.lr)
	print("Alpha: %.4g" % args.alpha)

	# Perform fitting and save model
	train_gnet_finetune(g, [patch_loaders, grid_loaders], args.lr, alpha=args.alpha, num_epochs=100,
		outfile=args.outfile, class_labels=class_names, accum_iters=5)
	
