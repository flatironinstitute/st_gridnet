import sys, os
import argparse
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("../..")
from datasets import PatchDataset, PatchGridDataset
from gridnet_patches import GridNet
from patch_classifier import densenet121, densenet_preprocess, patchcnn_simple
from training import train_gnet_atonce

from densenet import DenseNet



patch_size = 256
xform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(patch_size),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = "/mnt/ceph/users/adaly/datasets/aba_stdataset_20200212/"

img_train = os.path.join(data_dir, "imgs256_train")
lbl_train = os.path.join(data_dir, "lbls256_train")
grid_train = PatchGridDataset(img_train, lbl_train, transforms=xform)

img_val = os.path.join(data_dir, "imgs256_val")
lbl_val = os.path.join(data_dir, "lbls256_val")
grid_val = PatchGridDataset(img_val, lbl_val, transforms=xform)

class_names = ["Midbrain","Isocortex","Medulla","Striatum",
	"C. nuclei","C. cortex","Thalamus","O`lf. areas",
	"Cort. sub.","Pons","Pallidum","Hipp. form.","Hypothal."]
h_st, w_st = 32, 49

atonce_patch_limit = 32

'''

patch_size = 256
xform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(patch_size),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = "/mnt/ceph/users/adaly/datasets/maniatis_stdataset_20200714/"

img_train = os.path.join(data_dir, "imgs256_histmatch_train")
lbl_train = os.path.join(data_dir, "lbls256_histmatch_train")
grid_train = PatchGridDataset(img_train, lbl_train, transforms=xform)

img_val = os.path.join(data_dir, "imgs256_histmatch_val")
lbl_val = os.path.join(data_dir, "lbls256_histmatch_val")
grid_val = PatchGridDataset(img_val, lbl_val, transforms=xform)

class_names = ["Vent. Med. White", "Vent. Horn", "Vent. Lat. White", "Med. Gray", 
	"Dors. Horn", "Dors. Edge", "Med. Lat. White", "Vent. Edge", "Dors. Med. White",
	"Cent. Canal", "Lat. Edge"]
h_st, w_st = 35, 33

atonce_patch_limit = 32

'''

# Data Loaders
batch_size = 1
grid_loaders = {
	"train": DataLoader(grid_train, batch_size=batch_size, shuffle=False, pin_memory=True),
	"val": DataLoader(grid_val, batch_size=batch_size, shuffle=False, pin_memory=True)
}

# Model Instantiation
n_class = len(class_names)

f = DenseNet(num_classes=n_class, small_inputs=False, efficient=False,
	growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0)
g = GridNet(f, patch_shape=(3, patch_size, patch_size), grid_shape=(h_st, w_st), n_classes=n_class, 
	use_bn=False, atonce_patch_limit=atonce_patch_limit)

# Learning rate
lr = 0.001
alpha = 0.05
print("Learning Rate: %.4g" % lr)
print("Alpha: %.4g" % alpha)

# Perform fitting and save model
outfile = "../data/atonce_test"
train_gnet_atonce(g, grid_loaders, lr, alpha=alpha, num_epochs=100,
	outfile=outfile, class_labels=class_names)
