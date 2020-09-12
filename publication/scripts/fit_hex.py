import sys, os
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("../..")
from datasets import PatchDataset, PatchGridDataset
from densenet import DenseNet
from gridnet_patches import GridNetHex
from training import train_gnet_2stage, train_gnet_atonce

data_dir = "/mnt/ceph/users/adaly/datasets/maynard_patchdata_20200821/"

patch_size = 256
h_st, w_st = 78, 64
xform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(patch_size),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

atonce_patch_limit=32

img_train = os.path.join(data_dir, "imgs_train")
lbl_train = os.path.join(data_dir, "lbls_train")
patch_train = PatchDataset(img_train, lbl_train, xform)
grid_train = PatchGridDataset(img_train, lbl_train, xform)

img_val = os.path.join(data_dir, "imgs_val")
lbl_val = os.path.join(data_dir, "lbls_val")
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
outfile = "../data/gnethex_memdense_maynard"
train_gnet_2stage(g, [patch_loaders, grid_loaders], lr, alpha=alpha, num_epochs=100,
	outfile=outfile)

