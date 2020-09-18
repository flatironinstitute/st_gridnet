import os
import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = 100000000


############### MANIATIS DATASET ###############

# window_size: extract patches size proporitonally to image width, then down/upsample to patch_size.
# - Handles differing resolutions encountered in Maniatis data.
def extract_patches(img_file, st_coords, mapping_fn, patch_size, out_dir, array_name,
	window_size=None, ccdist=194):
	img = np.array(Image.open(img_file))
	ydim, xdim = img.shape[:2]

	# Create a sub-directory within the patch directory for the current array
	if not os.path.isdir(os.path.join(out_dir, array_name)):
		os.mkdir(os.path.join(out_dir, array_name))

	# Extract a separate jpg file for each patch and save in the subdirectory, with name indicating ST coordinates.
	for x,y in st_coords:
		pixel_x, pixel_y = mapping_fn([x,y], xdim, ydim, ccdist)

		if window_size is not None:
			# Extract a patch that is relatively sized to the current image
			w = int(window_size * xdim)
			patch = img[(pixel_y-w//2):(pixel_y+w//2), 
				(pixel_x-w//2):(pixel_x+w//2)]
			# Resize patch to match patch_size
			patch = np.array(Image.fromarray(patch).resize((patch_size, patch_size)))
		else:
			patch = img[(pixel_y-patch_size//2):(pixel_y+patch_size//2), 
				(pixel_x-patch_size//2):(pixel_x+patch_size//2)]

		spotfile = "%d_%d.jpg" % (int(np.rint(x)),int(np.rint(y)))
		Image.fromarray(patch).save(os.path.join(out_dir, array_name, spotfile))

def create_labelmat(st_coords, annots, st_dims, out_dir, array_name):
	lmat = np.zeros(st_dims).astype(np.int32)
	for c,a in zip(st_coords, annots):
		lmat[int(np.rint(c[1])),int(np.rint(c[0]))] = a+1 # Increment labels by 1 to reserve 0 for background.

	Image.fromarray(lmat).save(os.path.join(out_dir, array_name+".png"), format="PNG")

def st_to_pixel(c, xdim, ydim, ccdist=194):
	pixel_dim = float(ccdist)/(6200./xdim)	# Centroids on spot grid are 194 pixels apart in a 6200 pixel wide image.
	pixel_x = int(pixel_dim * (c[0]-1))
	pixel_y = int(pixel_dim * (c[1]-1))
	return [pixel_x, pixel_y]

def create_cartesian_dataset(wsi_files, annot_files, out_dir, patch_size=256, ccdist=194):
	patch_dir = os.path.join(out_dir, "imgs%d" % patch_size)
	label_dir = os.path.join(out_dir, "lbls%d" % patch_size)

	if not os.path.isdir(out_dir):
		os.mkdir(out_dir)
	if not os.path.isdir(patch_dir):
		os.mkdir(patch_dir)
	if not os.path.isdir(label_dir):
		os.mkdir(label_dir)

	st_dims = (35,33)

	# Extract st coordinates and patches for each file
	for img_file, annot_file in zip(wsi_files, annot_files):
		print("%s : %s ..." % (img_file, annot_file))

		slide = Path(annot_file).name.split(".")[0]

		df = pd.read_csv(annot_file, sep="\t", index_col=0)
		st_coords = [list(map(float, c.split("_"))) for c in df.columns.values]
		st_coords = np.array(st_coords)
		
		aar_names = df.index.values
		aar_inds = np.array(df).argmax(axis=0)
		fgd_inds = np.array(df).max(axis=0) > 0 # MUST ensure that all spots have annotations!!
		create_labelmat(st_coords[fgd_inds], aar_inds[fgd_inds], st_dims, label_dir, slide)

		extract_patches(img_file, st_coords[fgd_inds], st_to_pixel, 256, patch_dir, slide,
			ccdist=ccdist)


############### MAYNARD DATASET ###############

from visium_gridnet import VISIUM_H_ST, VISIUM_W_ST
from visium_gridnet import grid_from_wsi, to_hexagdly_label_tensor


def create_visium_dataset(wsi_files, annot_files, tpl_files, dest_dir, class_names, patch_size=256):
	if not os.path.isdir(dest_dir):
		os.mkdir(dest_dir)
	patch_dir = os.path.join(dest_dir, "imgs")
	label_dir = os.path.join(dest_dir, "lbls")
	for dname in [patch_dir, label_dir]:
		if not os.path.isdir(dname):
			os.mkdir(dname)

	for img_file, annot_file, tpl_file in zip(wsi_files, annot_files, tpl_files):
		print("%s : %s : %s ..." % (img_file, annot_file, tpl_file))

		# Extract name of current tissue
		tokens = Path(tpl_file).name.split("_tissue_positions_list")
		if len(tokens) > 0:
			slide = tokens[0]
		else:
			slide = tpl_file.split(".")[0]

		# Generate patch grid tensor (H_ST, W_ST, C, H_p, W_p) and label tensor (H_ST, W_ST)
		patch_grid = grid_from_wsi(img_file, tpl_file, patch_size)
		label_grid = to_hexagdly_label_tensor(annot_file, tpl_file, class_names)

		if not os.path.isdir(os.path.join(patch_dir, "%s" % slide)):
			os.mkdir(os.path.join(patch_dir, "%s" % slide))

		# Save all foreground patches as separate JPG files.
		for oddu_x in range(VISIUM_H_ST):
			for oddu_y in range(VISIUM_W_ST):
				if patch_grid[oddu_y, oddu_x].max() > 0:
					patch = patch_grid[oddu_y, oddu_x]

					patch = np.moveaxis(patch.data.numpy().astype(np.uint8), 0, 2)
					
					Image.fromarray(patch).save(
						os.path.join(patch_dir, "%s" % slide, "%d_%d.jpg" % (oddu_x, oddu_y)), "JPEG")
		
		Image.fromarray(label_grid.data.numpy().astype(np.int32)).save(
			os.path.join(label_dir, "%s.png" % slide), "PNG")

############### MAIN FUNCTION ###############

import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--outdir', type=str, required=True, help='Directory in which to save dataset')
	parser.add_argument('-i', '--imgs', nargs='+', required=True, help='Training set images')
	parser.add_argument('-a', '--annots', nargs='+', required=True, help='Annotation files corresponding to training images')
	parser.add_argument('-t', '--tpls', nargs='+', help='Tissue position listfiles corresponding to training images (required for Visium)')
	parser.add_argument('-c', '--class-names', nargs='+', help='Class names (will be inferred from first annotation file if not provided)')
	parser.add_argument('-p', '--patch-size', type=int, default=256, help='Size of patches (in pixels)')
	parser.add_argument('-d', '--ccdist', type=int, default=194, help='Center-center distance for Cartesian ST spots in an image with long dimension 6200px')
	args = parser.parse_args()

	assert len(args.imgs)==len(args.annots), "Number of images and annotation files do not match"

	if args.tpls is not None:
		assert len(args.imgs)==len(args.tpls), "Number of images and tissue position listfiles do not match"

		if args.class_names is None:
			df = pd.read_csv(args.annots[0], sep=",", header=0, names=['barcode', 'annotation'])
			args.class_names = list(np.unique(df['annotation']))

		create_visium_dataset(args.imgs, args.annots, args.tpls, args.outdir, args.class_names, args.patch_size)

	else:
		create_cartesian_dataset(args.imgs, args.annots, args.outdir, args.patch_size, args.ccdist)

