import os, sys
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("../..")
from datasets import PatchDataset, PatchGridDataset
from densenet import DenseNet
from gridnet_patches import GridNetHex
from training import train_gnet_2stage


# Read user arguments, specifying position in (folds x subfolds x repeats) job array
parser = argparse.ArgumentParser()
parser.add_argument('index', type=int, help="Index for current fold.")
parser.add_argument('-n', '--num-folds', type=int, default=6, help="Number of folds.")
parser.add_argument('-r', '--repeats', type=int, default=5, help="Number of repetitions of each fold (hyperparameter draws).")
parser.add_argument('-t', '--test-eval', action="store_true", help="Fit model on train+val, eval on test")
args = parser.parse_args()


# Map one-dimensional job array index to fold, sub-fold, and repeat index
folds = args.num_folds
sub_folds = args.num_folds-1
repetitions = args.repeats

assert args.index >= 0 and args.index < (folds * sub_folds * repetitions), "Index out of range for cross-validation settings."

# Optimal lr/alpha combinations for each fold discovered during nested cross-validation
best_lrs = [7.056e-4, 8.071e-4, 3.473e-4, 3.573e-4, 1.027e-4, 6.036e-4]
best_alphas = [7.951e-2, 2.993e-2, 6.747e-2, 9.957e-2, 9.744e-2, 3.896e-2]

# Load appropriate train and validation datasets for fold, sub-fold
data_dir = "/mnt/ceph/users/adaly/datasets/maynard_patchdata_20200821/"

if args.test_eval:
	test_fold = args.index
	outfile = "../data/gnethex_maynard_%d" % test_fold

	img_train = os.path.join(data_dir, "imgs_trainval_%dfold_%d" % (folds, test_fold))
	lbl_train = os.path.join(data_dir, "lbls_trainval_%dfold_%d" % (folds, test_fold))

	img_val = os.path.join(data_dir, "imgs_test_%dfold_%d" % (folds, test_fold))
	lbl_val = os.path.join(data_dir, "lbls_test_%dfold_%d" % (folds, test_fold))
else:
	test_fold = args.index // (sub_folds * repetitions)
	tmp = args.index - (test_fold * sub_folds * repetitions)
	rep = tmp % repetitions
	val_fold = tmp // repetitions
	outfile = "../data/gnethex_maynard_%d_%d" % (test_fold, val_fold)

	img_train = os.path.join(data_dir, "imgs_train_%dfold_%d_%d" % (folds, test_fold, val_fold))
	lbl_train = os.path.join(data_dir, "lbls_train_%dfold_%d_%d" % (folds, test_fold, val_fold))

	img_val = os.path.join(data_dir, "imgs_val_%dfold_%d_%d" % (folds, test_fold, val_fold))
	lbl_val = os.path.join(data_dir, "lbls_val_%dfold_%d_%d" % (folds, test_fold, val_fold))

patch_size = 256
h_st, w_st = 78, 64
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

batch_size=1
patch_loaders = {
	"train": DataLoader(patch_train, batch_size=32, shuffle=True, pin_memory=True),
	"val": DataLoader(patch_val, batch_size=32, shuffle=True, pin_memory=True)
}
grid_loaders = {
	"train": DataLoader(grid_train, batch_size=batch_size, pin_memory=True),
	"val": DataLoader(grid_val, batch_size=batch_size, pin_memory=True)
}


# Instantiate models and initialize parameters of DenseNet by transfer learning
class_names = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "White Matter"]
n_class = len(class_names)

# Best performing model from ABA
pth = "../data/aba_paramsweep/gnet_memdense_aba_lr0.0001857_alpha0.05618.pth"
f = DenseNet(num_classes=13, small_inputs=False, efficient=False,
	growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0)
if not torch.cuda.is_available():
	pdict = torch.load(pth, map_location=torch.device("cpu"))
else:
	pdict = torch.load(pth)
pretrain_dict = {k: pdict["patch_classifier.%s" % k] for k in f.state_dict().keys()}
f.load_state_dict(pretrain_dict)
f.classifier = torch.nn.Linear(1024, n_class)

# Note that grid dimensions are swapped due to rotation for HexagDLy indexing conventions!
g = GridNetHex(f, patch_shape=(3,patch_size,patch_size), grid_shape=(w_st, h_st), n_classes=n_class, 
	use_bn=False, atonce_patch_limit=atonce_patch_limit)


# Perform fitting and save model
if args.test_eval:
	lr = best_lrs[test_fold]
	alpha = best_alphas[test_fold]
else:
	lr = 10 ** (np.random.uniform(-4,-3))
	alpha = np.random.random() * 0.1
print("Learning Rate: %.4g" % lr)
print("Alpha: %.4g" % alpha)

train_gnet_2stage(g, [patch_loaders, grid_loaders], lr, alpha=alpha, num_epochs=50,
	outfile=outfile)
