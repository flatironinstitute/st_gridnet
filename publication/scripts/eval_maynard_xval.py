import sys, os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("../..")
from datasets import PatchGridDataset
from densenet import DenseNet
from gridnet_patches import GridNetHex
from utils import all_fgd_predictions, class_auroc


# Best-performing models on each fold (one per sub-fold, chosen from 5 hyperparameter draws)
best_models = [[
	"../data/gnethex_maynard_0_0_lr0.0001118_alpha0.06566.pth",
	"../data/gnethex_maynard_0_1_lr0.0006247_alpha0.08676.pth",
	"../data/gnethex_maynard_0_2_lr0.0008457_alpha0.09531.pth",
	"../data/gnethex_maynard_0_3_lr0.0009427_alpha0.06213.pth",
	"../data/gnethex_maynard_0_4_lr0.0007056_alpha0.07951.pth"
	]]
best_models.append([
	"../data/gnethex_maynard_1_0_lr0.0009532_alpha0.058.pth",
	"../data/gnethex_maynard_1_1_lr0.0005031_alpha0.09421.pth",
	"../data/gnethex_maynard_1_2_lr0.0008974_alpha0.08833.pth",
	"../data/gnethex_maynard_1_3_lr0.0007403_alpha0.08537.pth",
	"../data/gnethex_maynard_1_4_lr0.0008071_alpha0.02993.pth"
	])
best_models.append([
	"../data/gnethex_maynard_2_0_lr0.0001152_alpha0.07102.pth",
	"../data/gnethex_maynard_2_1_lr0.0007056_alpha0.07245.pth",
	"../data/gnethex_maynard_2_2_lr0.0007925_alpha0.07126.pth",
	"../data/gnethex_maynard_2_3_lr0.0005237_alpha0.08979.pth",
	"../data/gnethex_maynard_2_4_lr0.0003473_alpha0.06747.pth"
	])
best_models.append([
	"../data/gnethex_maynard_3_0_lr0.0009529_alpha0.08672.pth",
	"../data/gnethex_maynard_3_1_lr0.0007651_alpha0.07126.pth",
	"../data/gnethex_maynard_3_2_lr0.0005434_alpha0.06279.pth",
	"../data/gnethex_maynard_3_3_lr0.0009506_alpha0.06364.pth",
	"../data/gnethex_maynard_3_4_lr0.0003573_alpha0.09957.pth"
	])
best_models.append([
	"../data/gnethex_maynard_4_0_lr0.0005617_alpha0.06327.pth",
	"../data/gnethex_maynard_4_1_lr0.000763_alpha0.06248.pth",
	"../data/gnethex_maynard_4_2_lr0.0001072_alpha0.09744.pth",
	"../data/gnethex_maynard_4_3_lr0.0007509_alpha0.01142.pth",
	"../data/gnethex_maynard_4_4_lr0.0007165_alpha0.0683.pth"
	])
best_models.append([
	"../data/gnethex_maynard_5_0_lr0.0009475_alpha0.08156.pth",
	"../data/gnethex_maynard_5_1_lr0.0006576_alpha0.07579.pth",
	"../data/gnethex_maynard_5_2_lr0.0006036_alpha0.03896.pth",
	"../data/gnethex_maynard_5_3_lr0.0002143_alpha0.01768.pth",
	"../data/gnethex_maynard_5_4_lr0.0002234_alpha0.05769.pth"
	])


data_dir = "/mnt/ceph/users/adaly/datasets/maynard_patchdata_20200821/"

patch_size = 256
h_st, w_st = 78, 64
n_class = 7
xform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(patch_size),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
atonce_patch_limit = 32
batch_size=1

f = DenseNet(num_classes=n_class, small_inputs=False, efficient=False,
	growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0)
# Note that grid dimensions are swapped due to rotation for HexagDLy indexing conventions!
g = GridNetHex(f, patch_shape=(3,patch_size,patch_size), grid_shape=(w_st, h_st), n_classes=n_class, 
	use_bn=False, atonce_patch_limit=atonce_patch_limit)
g.eval()


# Store softmax predictions from which to make ensemble classifier
all_smax, all_patch_smax, all_true = [],[],[]

for fold in range(6):
	img_test = os.path.join(data_dir, "imgs_test_6fold_%d" % (fold))
	lbl_test = os.path.join(data_dir, "lbls_test_6fold_%d" % (fold))

	grid_test = PatchGridDataset(img_test, lbl_test, xform)

	fold_smax, fold_patch_smax = [],[]

	for pth in best_models[fold]:
		print(pth)

		# Import model parameters
		if torch.cuda.is_available():
			g.load_state_dict(torch.load(pth))
		else:
			g.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
		g.eval()

		# Run model on test set and store results
		grid_loader = DataLoader(grid_test, batch_size=batch_size, pin_memory=True)

		# Full model results:
		true, _, smax = all_fgd_predictions(grid_loader, g)
		fold_smax.append(smax)

		# Patch prediction results:
		true, _, smax = all_fgd_predictions(grid_loader, g, f_only=True)
		fold_patch_smax.append(smax)

	all_smax += list(np.array(fold_smax).mean(axis=0))
	all_patch_smax += list(np.array(fold_patch_smax).mean(axis=0))
	all_true += list(true)

all_smax = np.array(all_smax)
all_patch_smax = np.array(all_patch_smax)
all_true = np.array(all_true)

# Full model performance summary:
full_auroc = class_auroc(all_smax, all_true)
print("Consensus AUROC: {}".format("\t".join(list(map(str, full_auroc)))), flush=True)
print("Macro Avg: %.4f" % (full_auroc.mean()))

all_preds = all_smax.argmax(axis=1)
full_acc = float(sum(all_preds==all_true))/len(all_true)
print("Consensus Accuracy: %.4f" % full_acc)

full_patch_auroc = class_auroc(all_patch_smax, all_true)
print("Patch Consensus AUROC: {}".format("\t".join(list(map(str, full_patch_auroc)))), flush=True)
print("Patch Macro Avg: %.4f" % (full_patch_auroc.mean()))

all_patch_preds = all_patch_smax.argmax(axis=1)
full_patch_acc = float(sum(all_patch_preds==all_true))/len(all_true)
print("Consensus Accuracy: %.4f" % full_patch_acc)


