import numpy as np

import torch
import torch.nn.functional as F

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

# For each batch in dataloader, calculates model prediction.
# Returns a flattened, foreground-masked (true label > 0) list of spot predictions.
def all_fgd_predictions(dataloader, model, f_only=False):
	true_vals, pred_vals, pred_smax = [], [], []

	# GPU support
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	for x,y in dataloader:
		x = x.to(device)
		y = y.to(device)

		with torch.no_grad():
			if f_only:
				outputs = model.patch_predictions(x)
			else:		
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

# Find the distribution over number of misclassified neighbors for each misclassified spot.
def neighbor_error(true_lmat, pred_lmat, hexgrid=False):
    if hexgrid:
        raise NotImplementedError("Currently assume cartesian grid with 8-neighborhood")

    err_counts = np.zeros(9)
    ydim, xdim = true_lmat.shape
    for y in range(ydim):
        for x in range(xdim):
            # Find all foreground misclassifications
            if true_lmat[y,x]>0 and pred_lmat[y,x] != true_lmat[y,x]:
                misclass_neighbors = 0
                for j in [-1,0,1]:
                    for i in [-1,0,1]:
                        # Find all valid neighbors
                        if x+i>0 and x+i<xdim and y+j>0 and y+j<ydim and i+j!=0:
                            if pred_lmat[y+j,x+i] != true_lmat[y+j,x+i]:
                                misclass_neighbors += 1
                err_counts[misclass_neighbors] += 1
                
    return err_counts


############### Misclassification Density Plots ###############

def misclass_density(out_softmax, true):    
    ydim, xdim = true.shape
    
    mcd = np.zeros((ydim, xdim))
    
    for y in range(ydim):
        for x in range(xdim):
            # Only care about foreground patches
            if true[y,x] > 0:
                p_correct = out_softmax[true[y,x]-1, y, x]
                mcd[y][x] = 1-p_correct
    return mcd

def plot_class_boundaries(base_image, true):
    ydim, xdim = true.shape
    
    fig, ax = plt.subplots(1)
    plt.axis("off")
    
    # Mask out background spots and render over black background
    masked_image = np.ma.masked_where(true==0, base_image)
    bgd = ax.imshow(np.zeros_like(true), cmap="gray")
    fgd = ax.imshow(masked_image, cmap="plasma")
    
    xpix = 1.0/xdim
    ypix = 1.0/ydim
        
    for y in range(ydim):
        for x in range(xdim):
            for x_off in [-1, 1]:
                if x+x_off < 0 or x+x_off >= xdim:
                    continue
                if true[y,x] != true[y,x+x_off]:
                    ax.axvline(x=x+x_off/2, ymin=1-((y+1)*ypix), ymax=1-(y*ypix), c='w')
            for y_off in [-1, 1]:
                if y+y_off < 0 or y+y_off >= ydim:
                    continue
                if true[y,x] != true[y+y_off,x]:
                    ax.axhline(y=y+y_off/2, xmin=x*xpix, xmax=(x+1)*xpix, c='w')
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(fgd, cax=cax)
    cbar.set_label("Misclassification Probability")
    
    return fig


