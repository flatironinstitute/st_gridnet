''' Module designed to interface between Visium and GridNet, enabling:
	- Generation of training data for GridNet models from Visium images with associated 
	  Loupe annotations
	- Prediction of annotations for Visium arrays using pre-trained GridNet models.
'''

import sys, os
import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = 999999999

# GridNet imports
import torch
from torchvision import transforms
from src.densenet import DenseNet
from src.gridnet_patches import GridNetHex
from src.datasets import PatchDataset, PatchGridDataset

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


############### CONSTANTS ###############

VISIUM_H_ST = 78  # Visium arrays contain 78 rows (height)
VISIUM_W_ST = 64  # ...each row contains 64 spots (width)


############### HELPER FUNCTIONS ###############

def pseudo_hex_to_oddr(col, row):
	if col % 2 == 0:
		x = col/2
	else:
		x = (col-1)/2
	y = row
	return int(x), int(y)


############### VISIUM TO GRIDNET INPUTS ###############

''' Generates 5D tensor for input to GridNet from image file and spaceranger outputs.
	- fullres_imgfile: path to full-resolution image of tissue on Visium array.
	- tissue_positions_listfile: path to tissue_positions_list.csv, exported by spaceranger,
	  which maps Visium array indices to pixel coordinates in full-resolution image.
	- patch_size: size of image patches.
	- window_size: size of the window around each patch to be extracted.
'''
def grid_from_wsi(fullres_imgfile, tissue_positions_listfile, patch_size=256, window_size=256, 
	preprocess_xform=None):
	
	img = np.array(Image.open(fullres_imgfile))
	ydim, xdim = img.shape[:2]

	if window_size is None:
		w = patch_size
	elif isinstance(window_size, float):
		w = int(window_size * xdim)
	elif isinstance(window_size, int):
		w = window_size
	else:
		raise ValueError("Window size must be a float or int")

	# Pad image such that no patches extend beyond image boundaries
	img = np.pad(img, pad_width=((w//2, w//2), (w//2, w//2), (0,0)), mode='edge')

	df = pd.read_csv(tissue_positions_listfile, sep=",", header=None, 
		names=['barcode', 'in_tissue', 'array_row', 'array_col', 'px_row', 'px_col'])
	# Only consider spots that are within the tissue area.
	df = df[df['in_tissue']==1]

	# Visium uses a pseudo-hex indexing where the scale of the x-axis is doubled,
	# and every other row is shifted right by one (odd-right indexing).
	x_oddr, y_oddr = [],[]
	for i in range(len(df)):
		row = df.iloc[i]

		# Convert to standard integer indexing, which would lead to an implicit odd-right
		# addressed array if expressed in 2D (st_oddr).
		x,y = pseudo_hex_to_oddr(row['array_col'], row['array_row'])
		x_oddr.append(x)
		y_oddr.append(y)

	st_oddr = np.moveaxis(np.vstack((x_oddr, y_oddr)), 0, 1)
	# Rotate by 90 degrees to obtain an implicit odd-up addressed array, which is 
	# the input expected by HexagDLy. 
	st_oddu = np.rot90(st_oddr).astype(np.integer) # Indices into arraay for HexagDLy

	# Create a 5D tensor to store the image array, then populate with patches
	# extracted from the full-resolution image.
	img_tensor = torch.zeros((VISIUM_W_ST, VISIUM_H_ST, 3, patch_size, patch_size))
	for i in range(len(df)):
		x_ind, y_ind = st_oddu[:,i]
		x_px, y_px = df.iloc[i]['px_col'], df.iloc[i]['px_row']

		# Account for image padding
		x_px += w//2
		y_px += w//2

		patch = img[(y_px-w//2):(y_px+w//2), 
			(x_px-w//2):(x_px+w//2)]
		patch = np.array(Image.fromarray(patch).resize((patch_size, patch_size)))
		
		patch = torch.from_numpy(patch).permute(2,0,1)
		if preprocess_xform is not None:
			xf = transforms.Compose([
				transforms.ToPILImage(),
				transforms.ToTensor(),
				preprocess_xform
			])
			patch = xf(patch)

		if y_ind >= VISIUM_W_ST or x_ind > VISIUM_H_ST:
			print("Warning: row %d column %d outside bounds of Visium array" % (x_ind, y_ind))
			continue

		img_tensor[y_ind, x_ind] = patch

	return img_tensor.float()


############### LOUPE ANNOTATION <==> LABEL TENSORS ###############

def to_loupe_annots(label_tensor, tissue_positions_listfile, class_names, output_file):
	df = pd.read_csv(tissue_positions_listfile, sep=",", header=None, 
		names=['barcode', 'in_tissue', 'array_row', 'array_col', 'px_row', 'px_col'])
	# Only consider spots that are within the tissue area.
	df = df[df['in_tissue']==1]

	fh = open(output_file, "w+")
	fh.write("Barcode,Annotation\n")

	for i in range(len(df)):
		row = df.iloc[i]

		# Convert to standard integer indexing, which would lead to an implicit odd-right
		# addressed array if expressed in 2D (st_oddr).
		x,y = pseudo_hex_to_oddr(row['array_col'], row['array_row'])

		# Note that the label tensor will be rotated relative to native Visium representation
		# due to conventions of HexagDLy.
		annot = label_tensor[int(x),int(y)]

		# Foreground patches have annotations between 1 and N_class.
		if annot > 0:
			fh.write(row['barcode'] + "," + class_names[annot-1] + "\n")

	fh.close()

def to_hexagdly_label_tensor(loupe_annotfile, tissue_positions_listfile, class_names):
	df = pd.read_csv(tissue_positions_listfile, sep=",", header=None, 
		names=['barcode', 'in_tissue', 'array_row', 'array_col', 'px_row', 'px_col'])
	# Only consider spots that are within the tissue area.
	df = df[df['in_tissue']==1]

	af = pd.read_csv(loupe_annotfile, sep=",", header=0, names=['barcode', 'annotation'])

	# Note that grid dimensions are swapped due to rotation for HexagDLy indexing conventions!
	label_tensor = torch.zeros(VISIUM_W_ST, VISIUM_H_ST)
	for i in range(len(df)):
		row = df.iloc[i]

		# Skip un-annotated spots
		if not row['barcode'] in af['barcode']:
			continue

		a_row = af[af['barcode']==row['barcode']]
		annot_name = a_row['annotation'].iloc[0]
		try:
			annot_index = class_names.index(annot_name)
		except:
			raise ValueError("Annotation %s not in list of class names!" % annot_name)

		# Convert to standard integer indexing, which would lead to an implicit odd-right
		# addressed array if expressed in 2D (st_oddr).
		x,y = pseudo_hex_to_oddr(row['array_col'], row['array_row'])

		# Note that the label tensor will be rotated relative to native Visium representation
		# due to conventions of HexagDLy.
		if y >= VISIUM_H_ST or x >= VISIUM_W_ST:
			print("Warning: row %d column %d outside bounds of Visium array" % (y, x))
			continue

		label_tensor[int(x),int(y)] = annot_index+1  # Foreground indices are between 1 and N_class.

	return label_tensor.long()


############### GRIDNET LABEL PREDICTIONS ###############

def forward_pass(input_tensor, model_file, class_names, patch_size=256):
	n_class = len(class_names)

	atonce_patch_limit=32

	# Construct model and load parameters
	f = DenseNet(num_classes=n_class, small_inputs=False, efficient=False,
		growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0)
	# Note that grid dimensions are swapped due to rotation for HexagDLy indexing conventions!
	g = GridNetHex(f, patch_shape=(3,patch_size,patch_size), grid_shape=(VISIUM_W_ST, VISIUM_H_ST), 
		n_classes=n_class, use_bn=False, atonce_patch_limit=atonce_patch_limit)

	# Load parameters of fit model
	if torch.cuda.is_available():
		g.load_state_dict(torch.load(model_file))
	else:
		g.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

	g.eval()

	with torch.no_grad():
		y = g(torch.unsqueeze(input_tensor,0))[0]
		_, preds = torch.max(y, 0)

		for i in range(VISIUM_W_ST):
			for j in range(VISIUM_H_ST):
				if torch.max(x[i,j])==0:
					preds[i,j] = 0   # Zero-out background patches.
				else:
					preds[i,j] += 1  # Foreground patches between 1 and N_class.

	return preds


############### COMMAND LINE FUNCTIONALITY ###############

import argparse


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model-file', type=str, required=True, help="Path to model file")
	parser.add_argument('-i', '--img', type=str, required=True, help="Path to image file")
	parser.add_argument('-t', '--tpl', type=str, required=True, help="Path to tissue position list file")
	parser.add_argument('-c', '--class-names', nargs='+', required=True, help="Class names")
	parser.add_argument('-a', '--annot', type=str, help="Path for output annotation file")
	args = parser.parse_args()

	# Preprocessing applied to each patch:
	xform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	if args.annot is None:
		args.annot = Path(args.img).name.split('.')[0] + "_loupe.csv"

	x = grid_from_wsi(args.img, args.tpl, preprocess_xform=xform)
	y = forward_pass(x, args.model_file, args.class_names, patch_size=256)
	to_loupe_annots(y, args.tpl, args.class_names, args.annot)

