import sys, os
sys.path.append("..")
import patch_classifier as pc
import gridnet_patches as gn
from datasets import PatchDataset, PatchGridDataset
from utils import cmat_auroc, all_fgd_predictions

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import numpy as np
import pickle
from pathlib import Path

from resnet import resnet18

# Train GridNet in two stages:
# - Train f
# - Train g with parameters of f fixed
def train_gnet_2stage(model, dataloaders, lr, outfile=None, num_epochs=100, alpha=1.0):
	patch_loaders, grid_loaders = dataloaders

	# Define files in which to save output
	if outfile is not None:
		outfile = Path(outfile)
		model_file = outfile.parent / (outfile.stem + "_lr%.4g_alpha%.4g.pth" % (lr, alpha))
	else:
		model_file = None

	# Fit patch classifier f
	loss = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.patch_classifier.parameters(), lr=lr*alpha)
	best_pc, history = pc.train_model(model.patch_classifier, patch_loaders, loss, optimizer, 
		num_epochs=num_epochs)
	model.patch_classifier = best_pc

	# Fix parameters of the patch classifier.
	for param in model.patch_classifier.parameters():
		param.requires_grad = False

	# Fit corrector g
	loss = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.corrector.parameters(), lr=lr)
	best_gnet, history = gn.train_model(model, grid_loaders, loss, optimizer,
		outfile=model_file, num_epochs=num_epochs,
		finetune=False)

	return best_gnet, optimizer.state_dict()

# Train GridNet with end-to-end finetuning:
# - Train f and g as in 2stage
# - Train g and f 
def train_gnet_finetune(model, dataloaders, lr, alpha=1.0, num_epochs=100,
	outfile=None, class_labels=None,
	accum_iters=1):

	# Define files in which to save output
	if outfile is not None:
		outfile = Path(outfile)
		model_file = outfile.parent / (outfile.stem + "_lr%.4g_alpha%.4g.pth" % (lr, alpha))
	else:
		model_file = None

	# Train gnet in two-stage approach
	# Note: purposely using a single learning rate for both f and g for now
	gnet_init, g_opt_state = train_gnet_2stage(model, dataloaders, lr, num_epochs=num_epochs//2)

	# Allow backprop through patch classifier
	for param in gnet_init.patch_classifier.parameters():
		param.requires_grad = True

	# Fine-tune both networks
	grid_loaders = dataloaders[1]
	loss = nn.CrossEntropyLoss()

	# Train f with a re-initialized optimizer (and different learning rate)
	# Train g with an optimizer initialized to the final state of the pretraining optimizer.
	f_opt = torch.optim.Adam(model.patch_classifier.parameters(), lr=lr*alpha)
	g_opt = torch.optim.Adam(model.corrector.parameters(), lr=lr)
	g_opt.load_state_dict(g_opt_state)
	
	gnet_tune, history = gn.train_model(gnet_init, grid_loaders, loss, g_opt,
		f_opt=f_opt,
		outfile=model_file, num_epochs=num_epochs,
		finetune=True, accum_iters=accum_iters)

	# Log results of training
	print("Tuned GridNet Best Val. Acc.: %.4f" % np.max(history))
	
	cmat_counts, cmat_freq, auroc = cmat_auroc(grid_loaders['val'], gnet_tune, 
		model_file.stem, class_labels)

	print("Per-class AUROC:", "\t".join(list(map(str, auroc))))
	print("Macro Average AUROC: %.4f" % np.mean(auroc))

	return gnet_tune

# Train GridNet in the following manner:
# - Train f alone on patch data
# - Then alternate between training g with f fixed, and f with g fixed (both using grid data)
def train_gnet_interleave(model, dataloaders, lr, outfile=None, num_epochs=100, alpha=1.0):

	# Define files in which to save output
	if outfile is not None:
		outfile = Path(outfile)
		model_file = outfile.parent / (outfile.stem + "_model_%.6f_%.6f.pth" % (lr, alpha))
		hist_file = outfile.parent / (outfile.stem + "_trainhist_%.6f_%.6f.p" % (lr, alpha))
	else:
		model_file = None

	# Train gnet in two-stage approach
	gnet = train_gnet_2stage(model, dataloaders, lr, num_epochs=num_epochs)

	# At this point, parameters of f are fixed.

	# Toggle on/off state for gradient computation 
	def switch_training():
		for param in gnet.patch_classifier.parameters():
			param.requires_grad = not param.requires_grad
		for param in gnet.corrector.parameters():
			param.requires_grad = not param.requires_grad

	switch_epochs = 10 # Switch between training f and g every so often
	total_epochs = 0
	alpha = np.random.random() # Make lr for f training a fraction of lr for g training.
	optimizers = {
		'f': torch.optim.Adam(gnet.parameters(), lr=alpha*lr),
		'g': torch.optim.Adam(gnet.parameters(), lr=lr)
	}
	loss = nn.CrossEntropyLoss() # Same loss for both modes

	while total_epochs < num_epochs:
		for phase in ["f", "g"]:
			switch_training()
			print("Interleave Phase: %s" % phase)
			
			gnet, history = gn.train_model(gnet, dataloaders[1], loss,
				optimizers[phase], 
				num_epochs=switch_epochs, finetune=False, outfile=model_file)

			total_epochs += switch_epochs

	print("Alternate Training Best Val. Acc.: %.4f" % np.max(history))
	return gnet

# Train all the way through at once (end to end from beginning).
def train_gnet_atonce(model, grid_loaders, lr, outfile=None, num_epochs=100, 
	alpha=1.0, class_labels=None):

	# Define files in which to save output
	if outfile is not None:
		outfile = Path(outfile)
		model_file = outfile.parent / (outfile.stem + "_lr%.4g_alpha%.4g.pth" % (lr, alpha))
	else:
		model_file = None

	loss = nn.CrossEntropyLoss()

	# Train with different learning rates for patch classifier and corrector:
	optimizer = torch.optim.Adam([
		{'params': model.patch_classifier.parameters(), 'lr': alpha*lr}, 
		{'params': model.corrector.parameters()}
	], lr=lr)

	gnet, history = gn.train_model(model, grid_loaders, loss, optimizer,
		outfile=model_file, num_epochs=num_epochs,
		finetune=True)

	print("At-once Training Best Val. Acc.: %.4f" % np.max(history))

	cmat_counts, cmat_freq, auroc = cmat_auroc(grid_loaders['val'], gnet, 
		model_file.stem, class_labels)

	print("Per-class AUROC:", "\t".join(list(map(str, auroc))))
	print("Macro Average AUROC: %.4f" % np.mean(auroc))

	return gnet

# Pick up training of a model fit in the 2-stage scheme, and see if we can improve upon
#   validation accuracy with some setting for lr and alpha.
def train_gnet_refine(model, grid_loaders, lr, alpha=1.0, num_epochs=100,
	outfile=None, class_labels=None):

	# Define files in which to save output
	if outfile is not None:
		outfile = Path(outfile)
		model_file = outfile.parent / (outfile.stem + "_lr%.4g_alpha%.4g.pth" % (lr, alpha))
	else:
		model_file = None

	# Load pretrained model parameters
	pretrained_model = "training_output/gnet_simple_2stage_model_0.0004.pth"
	print("Refining model %s" % pretrained_model)

	if torch.cuda.is_available():
		model.load_state_dict(torch.load(pretrained_model))
	else:
		model.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu')))

	# Calculate validation performance of model before resuming training:
	for phase in grid_loaders.keys():
		true_vals, pred_vals, _ = all_fgd_predictions(grid_loaders[phase], model)
		pretrain_acc = float(np.sum(true_vals==pred_vals)) / len(true_vals)
		print("Pretrained Model %s Acc.: %.4f" % (phase, pretrain_acc))

	loss = nn.CrossEntropyLoss()

	# Train with different learning rates for patch classifier and corrector:
	optimizer = torch.optim.Adam([
		{'params': model.patch_classifier.parameters(), 'lr': alpha*lr}, 
		{'params': model.corrector.parameters()}
	], lr=lr)

	gnet, history = gn.train_model(model, grid_loaders, loss, optimizer,
		outfile=model_file, num_epochs=num_epochs,
		finetune=True)

	print("Refinement Best Val. Acc.: %.4f" % np.max(history))

	cmat_counts, cmat_freq, auroc = cmat_auroc(grid_loaders['val'], gnet, 
		model_file.stem, class_labels)

	print("Per-class AUROC:", "\t".join(list(map(str, auroc))))
	print("Macro Average AUROC: %.4f" % np.mean(auroc))

	return gnet


import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="GridNet Tissue Segmentation")
	parser.add_argument('imgdir', type=str, help="Path to directory containing training images.")
	parser.add_argument('lbldir', type=str, help="Path to directory containing training labels.")
	parser.add_argument('nclass', type=int, help="Number of foreground classes.")
	parser.add_argument('batchsize', type=int, help="Number of ST grids to use in a batch.")
	parser.add_argument('epochs', type=int, help="Number of training epochs.")
	parser.add_argument('-f', '--finetune', action="store_true", help="Train end-to-end after training f ang g separately")
	parser.add_argument('-i', '--interleave', action="store_true", help="Alternate between training f and g.")
	parser.add_argument('-a', '--atonce', action="store_true", help="Train both networks at once from the beginning.")
	parser.add_argument('-r', '--refine', action="store_true", help="Load a pretrained model and resume training (finetune) with new lr/alpha")
	parser.add_argument('-L', '--lr', type=float, help="Learning rate for g")
	parser.add_argument('-A', '--alpha', type=float, help="Fraction of lr for g to use training f")

	args = parser.parse_args()
	b = args.batchsize
	num_epochs = args.epochs
	n_classes = args.nclass
	class_labels = ["Midbrain", "Isocortex", "Medulla", "Striatum", 
		"Cerebellar nuclei", "Cerebellar cortex", "Thalamus", "Olfactory areas", 
		"Cortical subplate", "Pons", "Pallidum", "Hippocampal formation", "Hypothalamus"]

	# Find dimensions of ST grid for batch size calculation
	grid_dat = PatchGridDataset(args.imgdir, args.lbldir)
	x,y = next(iter(grid_dat))
	h_st, w_st = y.shape

	# Load fixed training and validation sets
	img_train = Path(args.imgdir).parent / (Path(args.imgdir).stem + "_train")
	img_val = Path(args.imgdir).parent / (Path(args.imgdir).stem + "_val")
	lbl_train = Path(args.lbldir).parent / (Path(args.lbldir).stem + "_train")
	lbl_val = Path(args.lbldir).parent / (Path(args.lbldir).stem + "_val")
	
	patch_train = PatchDataset(img_train, lbl_train)
	patch_val = PatchDataset(img_val, lbl_val)
	grid_train = PatchGridDataset(img_train, lbl_train)
	grid_val = PatchGridDataset(img_val, lbl_val)

	# Create dataloaders with equivalent batch sizes.
	patch_loaders = {
		"train": DataLoader(patch_train, batch_size=b*h_st*w_st, shuffle=True, pin_memory=True),
		"val": DataLoader(patch_val, batch_size=b*h_st*w_st, shuffle=True, pin_memory=True)
	}
	grid_loaders = {
		"train": DataLoader(grid_train, batch_size=b, shuffle=True, pin_memory=True),
		"val": DataLoader(grid_val, batch_size=b, shuffle=True, pin_memory=True)
	}

	# A simple f: ResNet18 with no BN layers and number of filters quartered
	clf = resnet18(13, thin=4)

	# A simple g: a few convolutional layers
	gnet = gn.GridNet(clf, x.shape[2:], y.shape, n_classes, use_bn=False)
	gnet.corrector = nn.Sequential(
			nn.Conv2d(n_classes, n_classes, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(n_classes, n_classes, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(n_classes, n_classes, 3, padding=1)
		)

	# Train with randomly sampled learning rate/alpha if not provided.
	if args.lr is None:
		lr = 10 ** (np.random.uniform(-4,-3))
	else:
		lr = args.lr
	if args.alpha is None:
		alpha = np.random.random() * 0.1
	else:
		alpha = args.alpha
	print("Learning Rate: %.4g" % lr)
	print("Alpha: %.4g" % alpha)
	
	if args.finetune:
		train_gnet_finetune(gnet, [patch_loaders, grid_loaders], lr, 
			outfile="../publication/data/gnet_simple_finetune", num_epochs=num_epochs,
			alpha=alpha)
	elif args.interleave:
		train_gnet_interleave(gnet, [patch_loaders, grid_loaders], lr,
			outfile="gnet_simple_interleave", num_epochs=num_epochs,
			alpha=alpha)
	elif args.atonce:
		train_gnet_atonce(gnet, grid_loaders, lr,
			outfile="../publication/data/gnet_simple_atonce", num_epochs=num_epochs,
			alpha=alpha, class_labels=class_labels)
	elif args.refine:
		train_gnet_refine(gnet, grid_loaders, lr, 
			alpha=alpha, class_labels=class_labels,
			outfile="gnet_simple_refine", num_epochs=num_epochs)
	else:
		train_gnet_2stage(gnet, [patch_loaders, grid_loaders], lr,
			outfile="../publication/data/gnet_simple_2stage", num_epochs=num_epochs,
			alpha=alpha)

