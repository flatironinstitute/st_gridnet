import os
import numpy as np
import pandas as pd

from PIL import Image
Image.MAX_IMAGE_PIXELS = 100000000


# window_size: extract patches size proporitonally to image width, then down/upsample to patch_size.
# - Handles differing resolutions encountered in Maniatis data.
def extract_patches(img_file, st_coords, mapping_fn, patch_size, out_dir, array_name,
	window_size=None):
	img = np.array(Image.open(img_file))
	ydim, xdim = img.shape[:2]

	# Create a sub-directory within the patch directory for the current array
	if not os.path.isdir(os.path.join(out_dir, array_name)):
		os.mkdir(os.path.join(out_dir, array_name))

	# Extract a separate jpg file for each patch and save in the subdirectory, with name indicating ST coordinates.
	for x,y in st_coords:
		pixel_x, pixel_y = mapping_fn([x,y], xdim, ydim)

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


############### MANIATIS DATASET ###############

def st_to_pixel(c, xdim, ydim):
	pixel_dim = 194./(6200./xdim)	# Centroids on spot grid are 194 pixels apart in a 6200 pixel wide image.
	pixel_x = int(pixel_dim * (c[0]-1))
	pixel_y = int(pixel_dim * (c[1]-1))
	return [pixel_x, pixel_y]

def create_maniatis_dataset(out_dir, patch_size=256):
	img_dir = os.path.expanduser("~/Dropbox (Simons Foundation)/Scaled/HE/")
	annot_dir = os.path.expanduser("~/Dropbox (Simons Foundation)/Covariates/")

	patch_dir = os.path.join(out_dir, "imgs%d" % patch_size)
	label_dir = os.path.join(out_dir, "lbls%d" % patch_size)
	if not os.path.isdir(patch_dir):
		os.mkdir(patch_dir)
	if not os.path.isdir(label_dir):
		os.mkdir(label_dir)
	st_dims = (35,33)

	# List of slides to be included:
	slides = []
	fh = open("../unsupervisedals/file_indices_mapping.tsv")
	for line in fh:
		_, annotfile = line.split("\t")
		slides.append(annotfile.split(".")[0])

	# Extract st coordinates and patches for each file
	for s in slides:
		print(s)

		img_file = os.path.join(img_dir, s+"_HE.jpg")
		annot_file = os.path.join(annot_dir, s+".tsv")

		df = pd.read_csv(annot_file, sep="\t", index_col=0)
		st_coords = [list(map(float, c.split("_"))) for c in df.columns.values]
		st_coords = np.array(st_coords)
		ws = 192./6200 * 0.75 # Fraction of image width (xdim) that corresponds to 150um
		
		aar_names = df.index.values
		aar_inds = np.array(df).argmax(axis=0)
		fgd_inds = np.array(df).max(axis=0) > 0 # MUST ensure that all spots have annotations!!
		create_labelmat(st_coords[fgd_inds], aar_inds[fgd_inds], st_dims, label_dir, s)

		extract_patches(img_file, st_coords[fgd_inds], st_to_pixel, 256, patch_dir, s,
			window_size=ws)


############### MAYNARD DATASET ###############

from visium_gridnet import grid_from_wsi, to_hexagdly_label_tensor

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

############### MAIN FUNCTION ###############

if __name__ == "__main__":
	create_maniatis_dataset(os.path.expanduser("~/Desktop/maniatis_stdataset_20200714/"))