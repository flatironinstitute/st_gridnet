import os
import csv
import numpy as np

import torch
from torch.utils.data import Dataset

''' Dataset class for spot-level (tissue type) classification based on expression of selected genes.
    Accepts:
    - count_dir: path to directory containing CSV files with rows [st_x, st_y, g_1, ... , g_G].
    - annot_dir: path to direcotry containing CSV files with rows [st_x, st_y, class_index].
    - metadata: path to CSV file containing rows mapping [count_dir_file, annot_dir_file].
    - st_dims: dimensions of ST array [h_st, w_st].
    
    Outputs:
    - count_grid: Float tensor of dimension [G, h_st, w_st]
    - label_grid: Long tensor of dimension [h_st, w_st]

    TODO: Add pre-processing option to normalize counts before return.
'''
class STCountDataset(Dataset):
    def __init__(self, count_dir, annot_dir, metadata, st_dims, 
                 metadata_delim='\t', count_delim=',', annot_delim='\t'):
        self.count_dir = count_dir
        self.annot_dir = annot_dir
        self.st_dims = st_dims
        self.count_delim = count_delim
        self.annot_delim = annot_delim
        self.mapping = self.read_metadata(metadata, delimiter=metadata_delim)
        
        self.fnames = []
        for f in os.listdir(count_dir):
            if f.endswith(".csv"):
                self.fnames.append(f)
    
    def read_metadata(self, fh, delimiter="\t"):
        csvfile = open(fh)
        reader = csv.reader(csvfile, delimiter=delimiter)

        # TODO: Make header optional?
        header = next(reader)

        mapping = {}
        for row in reader:
            mapping[row[0]] = row[1]
        return mapping
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        count_file = os.path.join(self.count_dir, self.fnames[idx])
        annot_file = os.path.join(self.annot_dir, self.mapping[self.fnames[idx]])
        
        spot_counts = np.loadtxt(count_file, delimiter=self.count_delim, skiprows=1)
        spot_annots = np.loadtxt(annot_file, delimiter=self.annot_delim, skiprows=1).astype("int32")
        
        g = spot_counts.shape[1]-2
        count_mat = np.zeros((g,) + self.st_dims )
        label_mat = np.zeros(self.st_dims)
        
        for a in spot_annots:
            label_mat[a[1], a[0]] = a[2] + 1 # 0 reserved as background class.
        for c in spot_counts:
            if label_mat[int(c[1]), int(c[0])] > 0:
                count_mat[:, int(c[1]), int(c[0])] = c[2:]
        
        return torch.from_numpy(count_mat).float(), torch.from_numpy(label_mat).long()