import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import numpy as np

class GridNet(nn.Module):
	def __init__(self, patch_classifier, patch_shape, grid_shape, n_classes, use_bn=True):
		super(GridNet, self).__init__()

		self.patch_shape = patch_shape
		self.grid_shape = grid_shape
		self.n_classes = n_classes
		self.patch_classifier = patch_classifier

		# Define Sequential model containing convolutional layers in global corrector.
		cnn_layers = []
		cnn_layers.append(nn.Conv2d(n_classes, n_classes, 3, padding=1))
		if use_bn:
			cnn_layers.append(nn.BatchNorm2d(n_classes))
		cnn_layers.append(nn.ReLU())
		cnn_layers.append(nn.Conv2d(n_classes, n_classes, 5, padding=2))
		if use_bn:
			cnn_layers.append(nn.BatchNorm2d(n_classes))
		cnn_layers.append(nn.ReLU())
		cnn_layers.append(nn.Conv2d(n_classes, n_classes, 5, padding=2))
		if use_bn:
			cnn_layers.append(nn.BatchNorm2d(n_classes))
		cnn_layers.append(nn.ReLU())
		cnn_layers.append(nn.Conv2d(n_classes, n_classes, 3, padding=1))
		self.corrector = nn.Sequential(*cnn_layers)

		# Define a constant vector to be returned by attempted classification of "background" patches
		self.bg = torch.zeros((1,n_classes))
		self.register_buffer("bg_const", self.bg) # Required for proper device handling with CUDA.

	# Wrapper function that calls patch classifier on foreground patches and returns constant values for background.
	def foreground_classifier(self, x):
		if torch.max(x) == 0:
			return self.bg_const
		else:
			return self.patch_classifier(x.unsqueeze(0))

	def patch_predictions(self, x):
		# Reshape input tensor to be of shape (batch_size * h_grid * w_grid, channels, h_patch, w_patch).
		patch_list = torch.reshape(x, (-1,)+self.patch_shape)

		patch_pred_list = torch.cat([self.foreground_classifier(p) for p in patch_list],0)

		patch_pred_grid = torch.reshape(patch_pred_list, (-1,)+self.grid_shape+(self.n_classes,))
		patch_pred_grid = patch_pred_grid.permute((0,3,1,2))

		return patch_pred_grid

	def forward(self, x):
		patch_pred_grid = self.patch_predictions(x)

		# Apply global corrector.
		corrected_grid = self.corrector(patch_pred_grid)
		
		return corrected_grid

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.BatchNorm2d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


import sys, os, time, copy
import argparse

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from patch_classifier import patchcnn_simple, densenet121, densenet_preprocess
from datasets import STPatchDataset, PatchGridDataset


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, outfile=None, finetune=False, accum_iters=1):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # GPU support
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
                model.eval()   # Set model to evaluate mode

            # Turn off batch normalization/dropout for patch classifier, but allow parameters to vary if finetuning
            model.patch_classifier.eval()
            if finetune and phase=='train':
                for param in model.patch_classifier.parameters():
                    param.requires_grad = True
            
            running_loss = 0.0
            running_corrects = 0
            running_foreground = 0

            # Iterate over data.
            for batch_ind, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs, then filter for foreground patches (label>0).
                    # Use only foreground patches in loss/accuracy calulcations.
                    outputs = model(inputs)
                    outputs = outputs.permute((0,2,3,1))
                    outputs = torch.reshape(outputs, (-1, outputs.shape[-1]))
                    labels = torch.reshape(labels, (-1,))
                    outputs = outputs[labels > 0]
                    labels = labels[labels > 0]
                    labels -= 1	# Foreground classes range between [1, N_CLASS].

                    loss = criterion(outputs, labels) / accum_iters
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        
                        if batch_ind % accum_iters == 0:
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

                if outfile is not None:
                    torch.save(model.state_dict(), outfile)
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)
    print('Best val Acc: {:4f}'.format(best_acc), flush=True)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def parse_args():
	parser = argparse.ArgumentParser(description="GridNet Tissue Segmentation")
	parser.add_argument('imgdir', type=str, help="Path to directory containing training images.")
	parser.add_argument('lbldir', type=str, help="Path to directory containing training labels.")
	parser.add_argument('-k', '--classes', type=int, default=2, help='Number of classes.')
	parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size.')
	parser.add_argument('-n', '--epochs', type=int, default=5, help='Number of training epochs.')
	parser.add_argument('-a', '--accum-iters', type=int, default=1, help='Perform optimizer step every "a" batches.')
	parser.add_argument('-o', '--output-file', type=str, default=None, help='Path to file in which to save best model.')
	parser.add_argument('-c', '--grad-checkpoints', type=int, default=0, help='Number of gradient checkpoints.')
	parser.add_argument('-p', '--patch-classifier', type=str, default=None, help='Path to pre-trained patch classifier.')
	parser.add_argument('-d', '--use-densenet', action="store_true", help='Use DenseNet121 architecture for patch classification.')
	parser.add_argument('-f', '--finetune', action="store_true", help='Fine-tune parameters of patch classifier.')
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	ACCUM_ITERS = args.accum_iters
	OUT_FILE = args.output_file
	EPOCHS = args.epochs
	BATCH_SIZE = args.batch_size
	CP = args.grad_checkpoints
	PC_PATH = args.patch_classifier
	USE_BN = (not args.finetune)

	# Old dataset formulation
	#grid_dataset = STPatchDataset(args.imgdir, args.lbldir, 128, 128)
	if args.use_densenet:
		xf = densenet_preprocess()
		grid_dataset = PatchGridDataset(args.imgdir, args.lbldir, xf)
	else:
		grid_dataset = PatchGridDataset(args.imgdir, args.lbldir)
	n_test = int(0.2 * len(grid_dataset))
	trainset, testset = random_split(grid_dataset, [len(grid_dataset)-n_test, n_test])

	train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
	test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
	dataloaders = {"train": train_loader, "val": test_loader}

	g, l = grid_dataset[0]
	h_st, w_st, c, h_patch, w_patch = g.shape
	fgd_classes = args.classes  
	
	if args.use_densenet:
		pc = densenet121(fgd_classes, pretrained=True, checkpoints=CP)
	else:
		pc = patchcnn_simple(h_patch, w_patch, c, fgd_classes, CP)
	gnet = GridNet(pc, patch_shape=(c, h_patch, w_patch), grid_shape=(h_st, w_st), n_classes=fgd_classes,
		use_bn=USE_BN)
	#gnet.apply(init_weights)

	# Load parameters of pre-trained patch classifier model, if provided.
	if PC_PATH is not None:
		if torch.cuda.is_available():
			print("CUDA available", flush=True)
			gnet.patch_classifier.load_state_dict(torch.load(PC_PATH))
		else:
			print("Fitting on CPU", flush=True)
			gnet.patch_classifier.load_state_dict(torch.load(PC_PATH, map_location=torch.device('cpu')))

		# For now, fix parameters of patch classifier before training corrector network.
		if not args.finetune:
			for param in gnet.patch_classifier.parameters():
				param.requires_grad = False

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(gnet.parameters(), lr=0.001)

	gnet_fit, hist = train_model(gnet, dataloaders, criterion, optimizer, 
		num_epochs=EPOCHS, outfile=OUT_FILE, finetune=args.finetune, accum_iters=ACCUM_ITERS)
	gnet_fit.to("cpu")

	# Visualize results from patch predictions, grid predictions on batches of train, test set
	gnet_fit.eval()
	gnet_fit.patch_classifier.eval()
	torch.set_grad_enabled(False)

	train_input, train_labels = next(iter(dataloaders["train"]))
	test_input, test_labels = next(iter(dataloaders["val"]))

	for batch, labels, name in [(train_input, train_labels, "train"), (test_input, test_labels, "test")]:

		patchpred = np.argmax(gnet_fit.patch_predictions(batch).data.numpy(), axis=1)
		gridpred = np.argmax(gnet_fit(batch).data.numpy(), axis=1)
		labels = labels.data.numpy()

		# Recall that output of gnet has dimensionality N_class - want to render foreground patches only, and as distinct from background (0)
		gridpred[labels==0] = 0
		gridpred[labels>0] += 1
		patchpred[labels==0] = 0
		patchpred[labels>0] += 1

		fig, ax = plt.subplots(BATCH_SIZE, 3, figsize=(9,3*BATCH_SIZE))

		for i in range(BATCH_SIZE):
			ax[i][0].imshow(patchpred[i], vmin=0, vmax=fgd_classes)
			ax[i][1].imshow(gridpred[i], vmin=0, vmax=fgd_classes)
			ax[i][2].imshow(labels[i], vmin=0, vmax=fgd_classes)

		ax[0][0].set_title("Patch Classifier")
		ax[0][1].set_title("GridNet")
		ax[0][2].set_title("True Labels")

		plt.savefig("gridnet_"+name+"_batch.png", format="png", dpi=300)
