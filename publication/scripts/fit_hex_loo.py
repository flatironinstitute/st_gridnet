import sys, os
import numpy as np
import argparse

from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("../..")
from datasets import PatchDataset, PatchGridDataset
from densenet import DenseNet
from gridnet_patches import GridNetHex
from training import train_gnet_2stage

# Construct datasets
parser = argparse.ArgumentParser()
parser.add_argument('loo', type=int, help="Index of array to leave out for validation")
parser.add_argument('lr', type=float, help="Learning rate")
parser.add_argument('alpha', type=float, help="Alpha")
args = parser.parse_args()

assert args.loo >= 0 and args.loo < 12, "Index must be within range of samples (12)."

data_dir = "/mnt/ceph/users/adaly/datasets/maynard_patchdata_20200821/"
img_train = os.path.join(data_dir, "imgs_train_%d" % args.loo)
lbl_train = os.path.join(data_dir, "lbls_train_%d" % args.loo)
img_val = os.path.join(data_dir, "imgs_val_%d" % args.loo)
lbl_val = os.path.join(data_dir, "lbls_val_%d" % args.loo)

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

class_names = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "White Matter"]
n_class = len(class_names)
f = DenseNet(num_classes=n_class, small_inputs=False, efficient=False,
	growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0)
# Note that grid dimensions are swapped due to rotation for HexagDLy indexing conventions!
g = GridNetHex(f, patch_shape=(3,patch_size,patch_size), grid_shape=(w_st, h_st), n_classes=n_class, 
	use_bn=False, atonce_patch_limit=atonce_patch_limit)

# Learning rate
print("Leave Out: %d" % args.loo)
print("Learning Rate: %.4g" % args.lr)
print("Alpha: %.4g" % args.alpha)

# Perform fitting and save model
outfile = "../data/gnethex_maynard%d" % args.loo
train_gnet_2stage(g, [patch_loaders, grid_loaders], args.lr, alpha=args.alpha, num_epochs=100,
	outfile=outfile)
