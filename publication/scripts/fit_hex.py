import sys, os
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("../..")
from src.datasets import PatchDataset, PatchGridDataset
from src.densenet import DenseNet
from src.gridnet_patches import GridNetHex
from src.training import train_gnet_2stage, train_gnet_atonce

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('imgtrain', type=str, help='Path to training set image directory')
parser.add_argument('lbltrain', type=str, help='Path to training set label directory')
parser.add_argument('imgval', type=str, help='Path to validation set image directory')
parser.add_argument('lblval', type=str, help='Path to validation set label directory')
parser.add_argument('-o', '--outfile', required=True, help='Path to save model output')
args = parser.parse_args()

patch_size = 256
h_st, w_st = 78, 64
xform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(patch_size),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

atonce_patch_limit=32

patch_train = PatchDataset(img_train, lbl_train, xform)
grid_train = PatchGridDataset(img_train, lbl_train, xform)

patch_val = PatchDataset(img_val, lbl_val, xform)
grid_val = PatchGridDataset(img_val, lbl_val, xform)

class_names = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "White Matter"]
n_class = len(class_names)

batch_size=1
patch_loaders = {
	"train": DataLoader(patch_train, batch_size=32, shuffle=True, pin_memory=True),
	"val": DataLoader(patch_val, batch_size=32, shuffle=True, pin_memory=True)
}
grid_loaders = {
	"train": DataLoader(grid_train, batch_size=batch_size, pin_memory=True),
	"val": DataLoader(grid_val, batch_size=batch_size, pin_memory=True)
}

f = DenseNet(num_classes=n_class, small_inputs=False, efficient=False,
	growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0)
# Note that grid dimensions are swapped due to rotation for HexagDLy indexing conventions!
g = GridNetHex(f, patch_shape=(3,patch_size,patch_size), grid_shape=(w_st, h_st), n_classes=n_class, 
	use_bn=False, atonce_patch_limit=atonce_patch_limit)

# Learning rate
lr = 10 ** (np.random.uniform(-4,-3))
alpha = np.random.random() * 0.1
print("Learning Rate: %.4g" % lr)
print("Alpha: %.4g" % alpha)

# Perform fitting and save model
train_gnet_2stage(g, [patch_loaders, grid_loaders], lr, alpha=alpha, num_epochs=100,
	outfile=args.outfile)

