''' Module designed to interface between Visium and GridNet, enabling:
	- Generation of training data for GridNet models from Visium images with associated 
	  Loupe annotations
	- Prediction of annotations for Visium arrays using pre-trained GridNet models.
'''

import sys, os
import numpy as np
import pandas as pd

from PIL import Image
Image.MAX_IMAGE_PIXELS = 999999999

# GridNet imports
import torch
from torchvision import transforms
sys.path.append(os.path.expanduser("~/Documents/Python/st_gridnet/"))
from densenet import DenseNet
from gridnet_patches import GridNetHex
from datasets import PatchDataset, PatchGridDataset

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
	return x, y


############### VISIUM TO GRIDNET INPUTS ###############

''' Generates 5D tensor for input to GridNet from image file and spaceranger outputs.
	- fullres_imgfile: path to full-resolution image of tissue on Visium array.
	- tissue_positions_listfile: path to tissue_positions_list.csv, exported by spaceranger,
	  which maps Visium array indices to pixel coordinates in full-resolution image.
	- patch_size: size of image patches to be extracted.
'''
def grid_from_wsi(fullres_imgfile, tissue_positions_listfile, patch_size=256, preprocess_xform=None):
	img = np.array(Image.open(fullres_imgfile))

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
	#img_array = np.zeros((VISIUM_W_ST, VISIUM_H_ST, patch_size, patch_size, 3))
	img_tensor = torch.zeros((VISIUM_W_ST, VISIUM_H_ST, 3, patch_size, patch_size))
	for i in range(len(df)):
		x_ind, y_ind = st_oddu[:,i]
		x_px, y_px = df.iloc[i]['px_col'], df.iloc[i]['px_row']

		patch = img[(y_px-patch_size//2):(y_px+patch_size//2), 
			(x_px-patch_size//2):(x_px+patch_size//2)]
		
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

		#img_array[y_ind, x_ind] = patch
		img_tensor[y_ind, x_ind] = patch

	#img_tensor = torch.from_numpy(img_array).permute((0,1,4,2,3))

	'''
	fig, ax = plt.subplots(1,3, figsize=(10,5))
	
	#ax[0].scatter(df['array_col'], df['array_row'])
	#ax[0].scatter(df['array_col'][:30], df['array_row'][:30], c='r')
	ax[0].scatter(st_oddr[:,0], st_oddr[:,1])
	ax[0].scatter(st_oddr[:30,0], st_oddr[:30,1], c='r')
	ax[0].invert_yaxis()
	ax[0].set_aspect('equal')

	#pos = np.moveaxis(np.vstack((df['array_col'], df['array_row'])), 0, 1)
	#pos = np.rot90(pos)
	#ax[1].scatter(pos[0,:], pos[1,:])
	#ax[1].scatter(pos[0,:][:30], pos[1,:][:30], c='r')
	ax[1].scatter(st_oddu[0,:], st_oddu[1,:])
	ax[1].scatter(st_oddu[0,:30], st_oddu[1,:30], c='r')
	ax[1].invert_yaxis()
	ax[1].set_aspect('equal')

	ax[2].imshow(img)
	ax[2].scatter(df['px_col'], df['px_row'])
	ax[2].set_aspect('equal')
	plt.show()
	'''

	return img_tensor.float()


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


############### CREATE MAYNARD GRIDNET TRAINING DATA ###############

def create_maynard_training_set(dest_dir):
	maynard_dir = os.path.expanduser("~/Documents/Splotch_projects/Splotch_DLPFC/data/")
	fullres_dir = os.path.expanduser("~/Desktop/Visium/Maynard_ImageData/")

	class_names = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"]
	
	slides = [151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 
		151673, 151674, 151675, 151676]

	patch_size = 256

	patch_dir = os.path.join(dest_dir, "imgs")
	label_dir = os.path.join(dest_dir, "lbls")
	for dname in [patch_dir, label_dir]:
		if not os.path.isdir(dname):
			os.mkdir(dname)

	for s in slides:
		print(s)

		annot_file = os.path.join(maynard_dir, "%d_loupe_annots.csv" % s)
		tpl_file = os.path.join(maynard_dir, "%d_tissue_positions_list.csv" % s)

		img_file = os.path.join(fullres_dir, "%d_full_image.tif" % s)

		patch_grid = grid_from_wsi(img_file, tpl_file, patch_size)
		label_grid = to_hexagdly_label_tensor(annot_file, tpl_file, class_names)

		# Render foreground spots and labels as a sanity check.
		#fig, ax = plt.subplots(1,2)
		#ax[0].imshow(patch_grid.data.numpy().max(axis=(2,3,4)), vmin=0, vmax=1)
		#ax[1].imshow(label_grid.data.numpy(), vmin=0, vmax=len(class_names))
		#plt.show()

		if not os.path.isdir(os.path.join(patch_dir, "%d" % s)):
			os.mkdir(os.path.join(patch_dir, "%d" % s))

		for oddu_x in range(VISIUM_H_ST):
			for oddu_y in range(VISIUM_W_ST):
				if patch_grid[oddu_y, oddu_x].max() > 0:
					patch = patch_grid[oddu_y, oddu_x]

					patch = np.moveaxis(patch.data.numpy().astype(np.uint8), 0, 2)
					
					Image.fromarray(patch).save(
						os.path.join(patch_dir, "%d" % s, "%d_%d.jpg" % (oddu_x, oddu_y)), "JPEG")
		
		Image.fromarray(label_grid.data.numpy().astype(np.int32)).save(
			os.path.join(label_dir, "%d.png" % s), "PNG")

def test_maynard_training_set(dest_dir):
	patch_dir = os.path.join(dest_dir, "imgs")
	label_dir = os.path.join(dest_dir, "lbls")

	pdat = PatchDataset(patch_dir, label_dir)
	print(len(pdat))

	x,y = next(iter(pdat))
	print(x.shape, x.min(), x.max())
	print(y)

	gdat = PatchGridDataset(patch_dir, label_dir)
	print(len(gdat))

	x,y = next(iter(gdat))
	print(x.shape, x.min(), x.max())
	print(y.shape, y.min(), y.max())


if __name__ == "__main__":
	sample = "V007-CGND-HRA-02746-A"
	slide_lot = "LV19B21-ID001_D"

	fullres_imgfile = "ALSFTD_Cortex/Visium_HE/%s.jpg" % slide_lot
	tissue_positions_listfile = "ALSFTD_Cortex/spaceranger_output/%s/spatial/tissue_positions_list.csv" % sample
	class_names = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "White Matter"]
	model_file = "ALSFTD_Cortex/gnethex_memdense_maynard_lr0.0006374_alpha0.09602.pth"

	# Preprocessing applied to each patch:
	xform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	x = grid_from_wsi(fullres_imgfile, tissue_positions_listfile, preprocess_xform=xform)
	y = forward_pass(x, model_file, class_names, patch_size=256)
	to_loupe_annots(y, tissue_positions_listfile, class_names, "%s_annots.csv" % sample)

	#dest_dir = os.path.expanduser("~/Documents/Splotch_projects/Splotch_DLPFC/data/maynard_patchdata_20200821/")
	#create_maynard_training_set(dest_dir)
	#test_maynard_training_set(dest_dir)


