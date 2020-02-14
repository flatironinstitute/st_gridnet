import h5py
import numpy as np
import pandas as pd
import glob
import numpy as np
import pandas as pd

def read_metadata(fh, delimiter="\t"):
    csvfile = open(fh)
    reader = csv.reader(csvfile, delimiter=delimiter)

    # TODO: Make header optional?
    header = next(reader)
    mapping = {}
    for row in reader:
        mapping[row[0]] = row[1]
    return mapping



def read_metadata(fh, delimiter="\t"):
    csvfile = open(fh)
    reader = csv.reader(csvfile, delimiter=delimiter)
    header = next(reader)
    mapping = {}
    for row in reader:
        mapping[row[0]] = row[1]
    return mapping


gene_file_names = glob.glob("/mnt/ceph/users/adaly/datasets/slide_count/genes/*.csv")
meta = read_metadata("/mnt/ceph/users/adaly/datasets/slide_count/metadata.csv")
annot_dir = "/mnt/ceph/users/adaly/datasets/slide_count/annotation/"

array_of_all_counts = np.empty([0, 23860])
tissue_mat = []
anno_file_mat = []

for count_mat in gene_file_names:
    file_name_count = count_mat.split("/")[-1]
    file_name_anno = annot_dir + meta[file_name_count]

    gene_file = np.loadtxt(count_mat, delimiter=",", skiprows=1)
    anno_file = np.loadtxt(file_name_anno, delimiter="\t", skiprows=1).astype("int32")
    gene_counts = gene_file[:,2:]

    spot_position = pd.DataFrame(data=gene_file[:,:2], columns=["x", "y"])
    anno_position = pd.DataFrame(data=anno_file, columns=["x", "y", "tissue"])
    tissue_class = pd.merge(spot_position, anno_position,  how='left', left_on=['x','y'], right_on = ['x','y'])
    tissue_class = tissue_class['tissue'].values
    anno_file_array = np.repeat(meta[file_name_count], len(tissue_class))

    array_of_all_counts = np.concatenate((array_of_all_counts, gene_counts), axis = 0)
    tissue_mat = np.append(tissue_mat, tissue_class)
    anno_file_mat = np.append(anno_file_mat, anno_file_array)

array_of_all_counts = array_of_all_counts[~np.isnan(tissue_mat),:]
anno_file_mat = anno_file_mat[~np.isnan(tissue_mat)]
tissue_mat = tissue_mat[~np.isnan(tissue_mat)]



hf = h5py.File('/mnt/home/thamamsy/data/mouse_spinal/data.h5', 'w')
hf.create_dataset('genes', data=array_of_all_counts)
hf.create_dataset('annotation', data=tissue_mat)
hf.close()

pd.DataFrame(anno_file_mat_final).to_csv("/mnt/home/thamamsy/data/mouse_spinal/meta_anno.csv")
