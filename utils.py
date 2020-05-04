import numpy as np

import torch.nn.functional as F

import matplotlib
from matplotlib import pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

# For each batch in dataloader, calculates model prediction.
# Returns a flattened, foreground-masked (true label > 0) list of spot predictions.
def all_fgd_predictions(dataloader, model):
	true_vals, pred_vals, pred_smax = [], [], []

	# GPU support
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	for x,y in dataloader:
		x = x.to(device)
		y = y.to(device)

		with torch.no_grad():		
			outputs = model(x)
			outputs = outputs.permute((0,2,3,1))
			outputs = torch.reshape(outputs, (-1, outputs.shape[-1]))
			labels = torch.reshape(y, (-1,))
			outputs = outputs[labels > 0]
			labels = labels[labels > 0] - 1 # Once background elimintated, re-scale between [0,N]

			outputs = outputs.cpu()
			labels = labels.cpu()
			y_fgd_true = labels.data.numpy()
			y_fgd_pred = torch.argmax(outputs, axis=1).data.numpy()
			y_fgd_smax = F.softmax(outputs, dim=1).data.numpy()

			true_vals.append(y_fgd_true)
			pred_vals.append(y_fgd_pred)
			pred_smax.append(y_fgd_smax)

	true_vals = np.concatenate(true_vals)
	pred_vals = np.concatenate(pred_vals)
	pred_smax = np.concatenate(pred_smax)

	return true_vals, pred_vals, pred_smax

# Accepts paired list of size (nsamples,) each containing 
def plot_confusion_matrix(y_true, y_pred, class_names, density):
	if np.min(y_true)>1:
		y_true -= 1
	if np.min(y_pred)>1:
		y_pred -= 1

	labels = range(0,len(class_names))
	cm_array = confusion_matrix(y_true,y_pred,labels=labels)
	
	fig, ax = plt.subplots(1, constrained_layout=True)
	if not density:
		cb = ax.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
		ax.set_title('Confusion matrix', fontsize=7)
		cbar = plt.colorbar(cb,fraction=0.046, pad=0.04)
		cbar.ax.tick_params(labelsize=7)
		cbar.set_label('Number of spots', rotation=270, labelpad=30, fontsize=7)
	else:
		denom = cm_array.sum(1,keepdims=True)
		denom = np.maximum(denom, np.ones_like(denom))
		cb = ax.imshow(cm_array/denom.astype(float),
			vmin=0,vmax=1,interpolation='nearest', cmap=plt.cm.Blues)
		ax.set_title('Normalized confusion matrix', fontsize=7)
		cbar = plt.colorbar(cb,fraction=0.046, pad=0.04)
		cbar.ax.tick_params(labelsize=7)
		cbar.set_label('Proportion of spots', rotation=270, labelpad=30, fontsize=7)

	xtick_marks = labels
	ytick_marks = labels
	ax.set_xticks(xtick_marks)
	ax.set_yticks(ytick_marks)
	ax.set_xticklabels(np.array(class_names),rotation=60,fontsize=7)
	ax.set_yticklabels(np.array(class_names),fontsize=7)
	ax.set_xlim([-0.5,len(class_names)-0.5])
	ax.set_ylim([len(class_names)-0.5,-0.5])
	ax.set_ylabel('True label',fontsize=7)
	ax.set_xlabel('Predicted label',fontsize=7)

	return fig

# Given a list of softmax predictions and corresponding "true" labels, outputs
#   the per-class AUROC as a numpy vector. 
# Macro Average AURUC can be computed as simple mean.
def class_auroc(smax, true):
	n_classes = smax.shape[1]
	auroc = np.zeros(n_classes)
	
	true_onehot = label_binarize(true, classes=list(range(n_classes)))

	for i in range(n_classes):
		fpr, tpr, _ = roc_curve(true_onehot[:,i], smax[:,i])
		auroc[i] = auc(fpr, tpr)

	return auroc

# Creates and saves plots of confusion matrices, and returns per-class AUROC.
def cmat_auroc(dataloader, model, name, class_labels=None):
	true, pred, smax = all_fgd_predictions(dataloader, model)

	cmat_counts = plot_confusion_matrix(true, pred, class_labels, density=False)
	plt.savefig(name+"_cmat_counts.png", format="PNG")

	cmat_freq = plot_confusion_matrix(true, pred, class_labels, density=True)
	plt.savefig(name+"_cmat_freq.png", format="PNG")
	
	# Calculate per-class AUROC
	auroc = class_auroc(smax, true)

	return cmat_counts, cmat_freq, auroc


import os
import torch

from datasets import PatchGridDataset
from gridnet_patches import GridNet
from patch_classifier import patchcnn_simple
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":
	imgdir = os.path.expanduser("~/Desktop/aba_stdataset_20200212/imgs128/")
	lbldir = os.path.expanduser("~/Desktop/aba_stdataset_20200212/lbls128/")
	dataset = PatchGridDataset(imgdir, lbldir)

	class_labels = ["Midbrain", "Isocortex", "Medulla", "Striatum", 
	"Cerebellar nuclei", "Cerebellar cortex", "Thalamus", "Olfactory areas", 
	"Cortical subplate", "Pons", "Pallidum", "Hippocampal formation", "Hypothalamus"]

	# Small test set (5 grids) to debug confusion matrix calculation
	testset, _ = random_split(dataset, [5, len(dataset)-5])

	#dl = DataLoader(dataset, batch_size=2)
	dl = DataLoader(testset, batch_size=1)

	pc = patchcnn_simple(128, 128, 3, 13)
	gnet = GridNet(pc, patch_shape=(3, 128, 128), grid_shape=(32, 49), n_classes=13)
	gnet.load_state_dict(torch.load("trained_models/gnet_simple_aba128.pth", 
		map_location=torch.device('cpu')))
	gnet.eval()

	cmat_counts, cmat_freq, auroc = cmat_auroc(dl, gnet, "aba128_gnsimple_test", class_labels)
	print(auroc)


