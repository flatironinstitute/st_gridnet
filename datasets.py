import os
import csv
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from PIL import Image

''' Dataset class for spot-level (tissue type) classification based on expression of selected genes.
    Accepts:
    - count_dir: path to directory containing CSV files with rows [st_x, st_y, g_1, ... , g_G].
    - annot_dir: path to direcotry containing CSV files with rows [st_x, st_y, class_index].
    - metadata: path to CSV file containing rows mapping [count_dir_file, annot_dir_file].
    - st_dims: dimensions of ST array [h_st, w_st].
    - normalize_spots: normalize each foreground spot so that gene expression vectors sum to 1.
    
    Outputs:
    - count_grid: Float tensor of dimension [G, h_st, w_st]
    - label_grid: Long tensor of dimension [h_st, w_st]

    TODO: Add option to normalize data to specified mean, std vectors.
'''
class STCountDataset(Dataset):
    def __init__(self, count_dir, annot_dir, metadata, st_dims, normalize_spots=False,
        metadata_delim='\t', count_delim=',', annot_delim='\t'):
        
        self.count_dir = count_dir
        self.annot_dir = annot_dir
        self.st_dims = st_dims
        self.count_delim = count_delim
        self.annot_delim = annot_delim
        self.mapping = self.read_metadata(metadata, delimiter=metadata_delim)
        self.normalize_spots = normalize_spots
        
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

                d = 1
                if self.normalize_spots:
                    d = np.sum(c[2:])
                count_mat[:, int(c[1]), int(c[0])] = c[2:]/d
                
        return torch.from_numpy(count_mat).float(), torch.from_numpy(label_mat).long()

''' Accepts paths to two directories containing training images (RBG) and label masks (grayscale).
    Corresponding images and labels should be identically named with the exception of the file format:
    .jpg for slide images, .png for label matrices.

    Label matrices should have dimensionality (H_ST, W_ST), and associated images should be (H_ST*P, W_ST*P, C)
    where P is the center-to-center distance between patches. 
    Preprocessing does not currently allow for margins on the image, so they should be cropped first for now.
'''
class STImageDataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        super(STImageDataset, self).__init__()
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir

        self.fnames = []
        for f in os.listdir(img_dir):
            if f.endswith(".jpg"):
                self.fnames.append(f.split(".")[0])

        # TODO: Add normalization to input preprocessing.
        self.xinput = Compose([ToTensor()])
        self.xlabel = Compose([ToTensor()])
            
    def __len__(self):
        return len(self.fnames)
    
    def get_lbl_tensor(self, idx):
        lbl_path = os.path.join(self.lbl_dir, self.fnames[idx]+".png")
        lbl = self.xlabel(Image.open(lbl_path))
        return torch.squeeze(lbl.long())
    
    def get_img_tensor(self, idx):
        img_path = os.path.join(self.img_dir, self.fnames[idx]+".jpg")
        img = self.xinput(Image.open(img_path))
        return img.float()
    
    def __getitem__(self, idx):
        return self.get_img_tensor(idx), self.get_lbl_tensor(idx)

''' Accepts directories of images, labels as above, but splits input images into tensors of image patches
    according to initialization parameters.
'''
class STPatchDataset(STImageDataset):
    def __init__(self, img_dir, lbl_dir, patch_size, patch_step):
        super(STPatchDataset, self).__init__(img_dir, lbl_dir)
        self.patch_size = patch_size
        self.patch_step = patch_step
    
    def __getitem__(self, idx):
        img_tensor = self.get_img_tensor(idx)
        
        grid = img_tensor.unfold(1, self.patch_size, self.patch_step)
        grid = grid.unfold(2, self.patch_size, self.patch_step)
        grid = grid.permute(1,2,0,3,4)
                
        return grid, self.get_lbl_tensor(idx)

###########################################################

import re
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


class PatchDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        super(PatchDataset, self).__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.patch_list = []
        self.coord_list = []

        rxp = re.compile("(\d+)_(\d+).jpg")

        # Look at each sub-directory, which each indicate a separate slide.
        dir_iter = os.walk(img_dir)
        top = next(dir_iter)
        for root, _, files in dir_iter:

            # Look for all correctly formatted image files within the subdirectories.
            for f in files:
                res = rxp.match(f)
                
                if res is not None:
                    self.patch_list.append(os.path.join(root, f))
                    x, y = int(res.groups()[0]), int(res.groups()[1])
                    self.coord_list.append([os.path.basename(root), x, y])

        self.preprocess = Compose([ToTensor()])

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        img = Image.open(self.patch_list[idx])
        img = self.preprocess(img)

        base, x, y = self.coord_list[idx]
        lbl = Image.open(os.path.join(self.label_dir, base+".png")).getpixel((x,y))

        return img.float(), torch.tensor(lbl).long()

class PatchGridDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        super(PatchGridDataset, self).__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.grid_list = []

        for f in os.listdir(label_dir):
            if f.endswith(".png"):
                s = f.split(".")[0]
                if s in os.listdir(img_dir):
                    self.grid_list.append(s)

        self.preprocess = Compose([ToTensor()])
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.grid_list)

    def __getitem__(self, idx):
        label_grid = Image.open(os.path.join(self.label_dir, self.grid_list[idx]+".png"))
        label_grid = torch.squeeze(self.totensor(label_grid))
        h_st, w_st = label_grid.shape

        patch_grid = None

        rxp = re.compile("(\d+)_(\d+).jpg")
        for f in os.listdir(os.path.join(self.img_dir, self.grid_list[idx])):
            res = rxp.match(f)
            if res is not None:
                x, y = int(res.groups()[0]), int(res.groups()[1])

                patch = Image.open(os.path.join(self.img_dir, self.grid_list[idx], f))
                patch = self.preprocess(patch)

                if patch_grid is None:
                    c,h,w = patch.shape
                    patch_grid = torch.zeros(h_st, w_st, c, h, w)

                patch_grid[y,x] = patch

        return patch_grid.float(), label_grid.long()

