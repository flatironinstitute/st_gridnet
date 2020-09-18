import os
import numpy as np

from matplotlib import pyplot as plt

from utils.datasets import PatchDataset, PatchGridDataset


############### ADJACENCY MATRIX ROUTINES ###############

def plot_connectivity_matrix(connect_mat, n_classes, class_names=None, class_counts=None):
	# Normalize connectivity matrix by the sum of each row
	for k in range(n_classes):
		connect_mat[k,:] = connect_mat[k,:]/sum(connect_mat[k,:])  
	
	# Plot the matrix, labeling all non-zero squares.
	fig = plt.figure(figsize=(8,8))
	plt.imshow(connect_mat, vmin=0.0, vmax=1.0)
	for i in range(n_classes):
		for j in range(n_classes):
			if connect_mat[i,j] > 0.0001:
				txt = "%.3f"%connect_mat[i, j]
			else:
				txt = ""
			text = plt.text(j, i, txt, ha="center", va="center", color="w", fontsize=8)
	
	if class_names is None:
		class_names = [""] * n_classes
	xlabels = ["%d. %s" % (i+1, c) for i,c in enumerate(class_names)]
	if class_counts is not None:
		ylabels = ["%d. %s (%d)" % (i+1, c, class_counts[i]) for i,c in enumerate(class_names)]
	else: 
		ylabels = xlabels
	plt.xticks(range(n_classes), xlabels, rotation=45)
	plt.yticks(range(n_classes), ylabels)

	fig.tight_layout()
	return fig

# Count the number of times class I neighbors class J (8-neighborhood)
def class_adjacency(grid_data, n_classes, class_names=None):
	connect_mat = np.zeros((n_classes, n_classes))
	class_counts = np.zeros(n_classes, dtype=np.int32)
	
	for _,l_tensor in grid_data:
		l = l_tensor.data.numpy()
		ydim, xdim = l.shape
		for y in range(ydim):
			for x in range(xdim):
				c1 = l[y,x]
				if c1 > 0:
					class_counts[c1-1] += 1

					for i in [-1,0,1]:
						for j in [-1,0,1]:
							if x+i>0 and x+i<xdim and y+j>0 and y+j<ydim:
								c2 = l[y+j, x+i]
								if c2 > 0 and i+j != 0:
									connect_mat[c1-1,c2-1] += 1
									
	return plot_connectivity_matrix(connect_mat, n_classes, class_names, class_counts)

# Count the number of times class I neighbors class J (6-neighborood, odd-up indexing)
def class_adjacency_hex(grid_data, n_classes, class_names=None):
	connect_mat = np.zeros((n_classes, n_classes))
	class_counts = np.zeros(n_classes, dtype=np.int32)

	for _,l_tensor in grid_data:
		l = l_tensor.data.numpy()
		ydim, xdim = l.shape
		for y in range(ydim):
			for x in range(xdim):
				c1 = l[y,x]
				if c1 > 0:
					if c1 > 0:
						class_counts[c1-1] += 1

					# If x is even, neighbors are (x, y+/-1), (x+/-1, y), (x+/-1, y-1)
					if x % 2 == 0:
						nset = [(0,1), (0,-1), (1,0), (-1,0), (1,-1), (-1,-1)]
					# If x is odd, neighbors are (x, y+/-1), (x+/-1, y), (x+/-1, y+1)
					else:
						nset = [(0,1), (0,-1), (1,0), (-1,0), (1,+1), (-1,+1)]
					for (dx,dy) in nset:
						if x+dx>0 and x+dx<xdim and y+dy>0 and y+dy<ydim:
							c2 = l[y+dy, x+dx]
							if c2 > 0:
								connect_mat[c1-1, c2-1] += 1

	return plot_connectivity_matrix(connect_mat, n_classes, class_names, class_counts)
					

############### LABEL RENDERING ROUTINES ###############

# Find an array that contains patches of all classes.
# If such an array doesn't exist, choose one with maximal number of classes.
def select_representative_image(grid_data, n_classes, class_names=None):
	max_classes = 0
	best_array = None

	for x,y in grid_data:
		classes_present = len(np.unique(y.data.numpy()))
		
		if classes_present == n_classes+1:
			return y
		else:
			if classes_present > max_classes:
				max_classes = classes_present
				best_array = y

	return best_array

def plot_labels(grid_data, n_classes, class_names=None, selected_index=None):
	if selected_index:
		_, label_mat = grid_data[selected_index]
	else:
		label_mat = select_representative_image(grid_data, n_classes, class_names)

	ydim, xdim = label_mat.shape
	fig = plt.figure(figsize=(8, 8*(ydim/xdim)))
	ax = fig.add_subplot(111)

	cmap = plt.get_cmap("hsv")
	ax.set_prop_cycle(color=[cmap(float(i)/n_classes) for i in range(n_classes)])

	if class_names is None:
		class_names = ["%d" % (i+1) for i in range(n_classes)]

	for c in range(n_classes):
		ys,xs = np.where(label_mat.data.numpy()==(c+1))
		ax.scatter(xs, ys, label=class_names[c])
	ax.set_aspect('equal')
	#ax.set_xlim(0,xdim)
	#ax.set_ylim(0,ydim)
	ax.invert_yaxis()

	plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
	plt.tight_layout()
	return fig

# Assumes label_mat obeys HexagDLy addressing scheme
def plot_labels_hex(grid_data, n_classes, class_names=None, selected_index=None):
	if selected_index:
		_, label_mat = grid_data[selected_index]
	else:
		label_mat = select_representative_image(grid_data, n_classes, class_names)

	ydim, xdim = label_mat.shape
	fig = plt.figure(figsize=(8, 8), constrained_layout=True)
	ax = fig.add_subplot(111)

	cmap = plt.get_cmap("hsv")
	ax.set_prop_cycle(color=[cmap(float(i)/n_classes) for i in range(n_classes)])

	if class_names is None:
		class_names = ["%d" % (i+1) for i in range(n_classes)]

	for c in range(n_classes):
		ys,xs = np.where(label_mat.data.numpy()==(c+1))

		# To revert to cartesian coordinates, shift odd rows down, scale, and rotate
		cart_coords = np.zeros((len(xs),2))
		for i,(x,y) in enumerate(zip(xs, ys)):
			if x % 2 == 1:
				cart_coords[i,:] = [x, y+0.5]
			else:
				cart_coords[i,:] = [x, y]
		cart_coords[:,0] = cart_coords[:,0] * np.sqrt(3)/2
		c_x, c_y = np.rot90(cart_coords, k=-3)

		ax.scatter(c_x, c_y, label=class_names[c], s=8.0)
	ax.invert_yaxis()
	ax.set_aspect('equal')

	plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
	return fig


############### REPRESENTATIVE PATCH ROUTINES ###############

def plot_class_examples(patch_dataset, n_classes, class_names=None, n_samples=3):
	# Find representative samples for each class
	patches = {}
	count = 0
	for x,y in patch_dataset:
		c_ind = int(y)
		if c_ind not in patches.keys():
			patches[c_ind] = []
		if len(patches[c_ind]) < n_samples:
			patches[c_ind].append(np.moveaxis(x.data.numpy(),0,-1))
			count += 1

		if count == n_classes * n_samples:
			break

	if class_names is None:
		class_names = ["%d" % (i+1) for i in range(n_classes)]

	n_rows = int(np.ceil(n_classes/2))
	n_cols = 2*n_samples

	fig = plt.figure()
	count = 1
	for c in range(n_classes):
		for n in range(n_samples):
			ax = fig.add_subplot(n_rows, n_cols, count)
			ax.imshow(patches[c+1][n])
			ax.set_title(class_names[c], fontsize=8)
			ax.axis('off')

			count += 1
	
	plt.tight_layout()
	return fig

