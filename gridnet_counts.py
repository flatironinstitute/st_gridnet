''' Aidan Daly - 01/27/2020

	Create a CNN operating on ST count data (H_ST, W_ST, G) to predict tissue type.

	Works similarly to second stage of GridNet, and hopefully first step towards an integrated approach
	to tissue registration using HE & ST data.
'''

########## MODEL DEFINITION ##########

import torch
import torch.nn as nn

# Very bad name - we will think of something better once we figure ou thow everything plays together.
class CountGridNet(nn.Module):

	# grid_shape: [H_ST, W_ST]
	# n_classes: number of tissue classes + 1 (for "background")
	def __init__(self, n_genes, n_classes):
		super(CountGridNet, self).__init__()

		self.n_genes = n_genes
		self.n_classes = n_classes

		# Define a Sequential model, which is just a wrapper function for the successive application of
		#   a series of operations -- e.g, convolution, batchnorm, activation... -- on the input.
		# TODO: We may want to look into modifying existing architectures (e.g., ResNet) in lieu of this
		#   custom approach, although they are designed for larger images (224x224, as opposed to 33x35).
		layers = []
		# In the first layer, the number of genes is the input channels.
		# Padding should be equal to (ksize-1)/2 to obtain output of same size as input.
		layers.append(nn.Conv2d(n_genes, 512, 3, padding=1)) 
		layers.append(nn.BatchNorm2d(512))
		layers.append(nn.ReLU())
		layers.append(nn.Conv2d(512, 1024, 3, padding=1))
		layers.append(nn.BatchNorm2d(1024))
		layers.append(nn.ReLU())
		layers.append(nn.Conv2d(1024, 512, 3, padding=1))
		layers.append(nn.BatchNorm2d(512))
		layers.append(nn.ReLU())
		layers.append(nn.Conv2d(512, 256, 3, padding=1))
		layers.append(nn.BatchNorm2d(256))
		layers.append(nn.ReLU())
		layers.append(nn.Conv2d(256, n_classes, 3, padding=1))
		self.cnn = nn.Sequential(*layers)

		# NOTE: PyTorch computes SoftMax inside the cross-entropy loss function, so models should return logits
		# (this was a pretty strange/insidious decision that gave me a lot of weird results early on).

	# Method that must be overriden to produce forward simulations.
	# Can be called as: model(input)
	def forward(self, x):
		return self.cnn(x)

########## MODEL TRAINING ##########

import time, copy

# Train model, only allowing for foreground patches to contribute to error.
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, accum_size=1):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # GPU support - if GPU is detected, model (and all subsequent inputs to the model) must be explicitly
    #   moved from CPU memory to GPU memory.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
        print('-' * 10, flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode (batch normalization, dropout turned off)

            running_loss = 0.0
            running_corrects = 0
            running_foreground = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs, then filter for foreground patches (label>0).
                    # Use only foreground patches in loss/accuracy calulcations.
                    outputs = model(inputs)
                    outputs = outputs.permute((0,2,3,1)) # Move softmax classification to last dimension.
                    outputs = torch.reshape(outputs, (-1, outputs.shape[-1])) # Flatten batch & ST dims - [batchsize*H_ST*W_ST, N_class].
                    labels = torch.reshape(labels, (-1,)) # Flatten label matrix so that dimensions match up.
                    outputs = outputs[labels > 0]
                    labels = labels[labels > 0]

                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                    	frac_loss = loss / accum_size
                    	frac_loss.backward()

                    	if (i+1) % accum_size == 0:
	                        optimizer.step()
	                        optimizer.zero_grad()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_foreground += len(labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / running_foreground

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)
    print('Best val Acc: {:4f}'.format(best_acc), flush=True)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


########## MAIN FUNCTION ##########

import sys
import argparse
import torch.optim as optim

from datasets import STCountDataset
from torch.utils.data import DataLoader

def parse_args():
	parser = argparse.ArgumentParser(description="Registration of ST spots to AAR framework based on counts of selected genes.")
	parser.add_argument('countdir', type=str, help="Path to directory containing numpy archives of ST count matrices.")
	parser.add_argument('labeldir', type=str, help="Path to directory containing numpy archives of ST grid labelings.")
	parser.add_argument('metadata', type=str, help="Path to metadata file mapping count files to label files.")
	parser.add_argument('hst', type=int, help="Height of ST array (number of spots).")
	parser.add_argument('wst', type=int, help="Width of ST array (number of spots).")
	parser.add_argument('nclass', type=int, help="Number of tissue classes (excluding background).")
	parser.add_argument('-n', '--epochs', type=int, default=10, help="Number of training epochs.")
	parser.add_argument('-b', '--batch-size', type=int, default=5, help="Size of training batches.")
	parser.add_argument('-a', '--accum-size', type=int, default=1, help='Perform optimizer step every "a" batches.')
	parser.add_argument('-o', '--output-file', type=str, default=None, help="Path to save trained model.")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	st_count_dataset = STCountDataset(args.countdir, args.labeldir, args.metadata, (args.hst, args.wst), 
		normalize_spots=True)

	# Create train-test split, then declare DataLoaders, which handle batching.
	n_val = int(0.2*len(st_count_dataset))
	trainset, valset = torch.utils.data.random_split(st_count_dataset, [len(st_count_dataset)-n_val, n_val])
	x,y = trainset[0]
	G,_,_ = x.shape

	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True)
	data_loaders = {"train":train_loader, "val":val_loader}

	model = CountGridNet(G, args.nclass+1)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	best_model, history = train_model(model, data_loaders, criterion, optimizer, 
		num_epochs=args.epochs, accum_size=args.accum_size)

	if args.output_file is not None:
		torch.save(best_model.state_dict(), args.output_file)

