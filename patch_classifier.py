import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from torchvision import models, transforms

import copy


''' Custom CNN architecture that achieves 70% test accuracy on ABA brain dataset
'''
class PatchCNN(nn.Module):
	def __init__(self, patch_shape, n_classes, conv_layers, fc_layers, pksize=3, poolsize=2, checkpoints=0):
		super(PatchCNN, self).__init__()

		assert isinstance(checkpoints, int) and checkpoints >= 0, "Number of checkpoints must be non-negative integer" 
		self.checkpoints = checkpoints

		cnn_layers = []
		cnn_layers.append(nn.Conv2d(patch_shape[0], conv_layers[0], pksize, padding=(pksize-1)//2))
		cnn_layers.append(nn.BatchNorm2d(conv_layers[0]))
		cnn_layers.append(nn.ReLU())

		n_pool = 0
		for i in range(1,len(conv_layers)):
			cnn_layers.append(nn.Conv2d(conv_layers[i-1], conv_layers[i], pksize, padding=(pksize-1)//2))
			cnn_layers.append(nn.BatchNorm2d(conv_layers[i]))
			cnn_layers.append(nn.ReLU())

			if (i+1) % 2 == 0:
				cnn_layers.append(nn.MaxPool2d(poolsize))
				n_pool += 1

		self.cnn = nn.Sequential(*cnn_layers)

		fcn_layers = []
		latent_size = patch_shape[1]//(poolsize**n_pool) * patch_shape[2]//(poolsize**n_pool) * conv_layers[-1]
		fcn_layers.append(nn.Linear(latent_size, fc_layers[0]))
		fcn_layers.append(nn.ReLU())
		fcn_layers.append(nn.Dropout(0.5))

		for i in range(1,len(fc_layers)):
			fcn_layers.append(nn.Linear(fc_layers[i-1], fc_layers[i]))
			fcn_layers.append(nn.ReLU())
			fcn_layers.append(nn.Dropout(0.5))

		# NOTE: nn.CrossEntropyLoss applies Softmax internally, so model should output raw logits.
		# Recall that model output will need to be passed through softmax prior to input to CRF, etc.
		fcn_layers.append(nn.Linear(fc_layers[-1], n_classes))

		self.fcn = nn.Sequential(*fcn_layers)

	def forward(self, x):
		if self.checkpoints == 0:
			out = self.cnn(x)
		else:
			# Gradient checkpointing breaks unless at least one input has requires_grad=True.
			# Thus, input must pass through at least one layer before CP (unless we develop a clevel hack).
			# NOTE: This only solves breaking of gradient tape if network is trained end-to-end.
			out = self.cnn[0](x)
			out = checkpoint_sequential(self.cnn[1:], self.checkpoints, out)

		out = out.reshape(out.size(0), -1)
		out = self.fcn(out)

		return out

def patchcnn_simple(h_patch, w_patch, c, n_classes, checkpoints=0):
	cl = [16,32,64,128,256,128,64,8]
	fl = [100,50]
	pc = PatchCNN(patch_shape=(c, h_patch, w_patch), n_classes=n_classes, conv_layers=cl, fc_layers=fl, 
		checkpoints=checkpoints)
	return pc

''' Modified DenseNet with optional pretraining on ImageNet.
	
	Pre-trained models expect input to be transformed according to the following:
	- Shape (3, H, W) where H, W > 224
	- Loaded into range [0,1]
	- Normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
'''
class DenseNet121(nn.Module):
	def __init__(self, n_classes, pretrained=False, checkpoints=0):
		super(DenseNet121, self).__init__()
		self.n_classes = n_classes
		self.checkpoints = checkpoints

		self.dnet = models.densenet121(pretrained)
		self.dnet.classifier = nn.Linear(1024, n_classes)

	# Allows loading state dict of EITHER:
	# - Model instatiated with this class.
	# - Torchvision DenseNet model.
	def load_state_dict(self, state_dict, strict=True):
		sd2 = copy.deepcopy(state_dict)
		for key in state_dict:
			# If parameters start like "dnet.features", state dict is from an instance of this class.
			# For some reason, the modified classifier is saved directly as "classifier", while "dnet.classifier"
			#  retains parameters for default 1000-state classification layer. Manually ignore the latter.
			if 'dnet' in key:
				if 'classifier' in key:
					sd2.pop(key)
				else:
					sd2[key.split('dnet.')[1]] = sd2.pop(key)

		return self.dnet.load_state_dict(sd2, strict)

	def forward(self, x):
		if self.checkpoints == 0:
			features = self.dnet.features(x)
		else:
			# Gradient checkpointing breaks unless at least one input has requires_grad=True.
			# Thus, input must pass through at least one layer before CP (unless we develop a clever hack).
			# NOTE: This only solves breaking of gradient tape if network is trained end-to-end.
			features = self.dnet.features[0](x)
			features = checkpoint_sequential(self.dnet.features[1:], self.checkpoints, features)

		out = F.relu(features, inplace=True)
		out = F.adaptive_avg_pool2d(out, (1, 1))
		out = torch.flatten(out, 1)
		out = self.dnet.classifier(out)

		return out

#def densenet121(n_classes, pretrained=False):
#	dnet = models.densenet121(pretrained)
#	dnet.classifier = nn.Linear(1024, n_classes)
#	return dnet

def densenet121(n_classes, pretrained=False, checkpoints=0):
	return DenseNet121(n_classes, pretrained, checkpoints)

# Helper method - returns a preprocessing transform for input to DenseNet which can be supplied as an optional 
#  argument to PatchDataset or PatchGridDataset.
def densenet_preprocess():
	preprocess = transforms.Compose([
    	transforms.Resize(256),
    	transforms.CenterCrop(224),
    	transforms.ToTensor(),
    	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	return preprocess 

#################### MAIN FUNCTION FOR PATCH CLASSIFIER TRAINING ##########################

import time, copy
import argparse
from datasets import PatchDataset
from torch.utils.data import DataLoader, random_split

def train_model(model, dataloaders, criterion, optimizer, num_epochs, outfile=None):
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

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels -= 1	# Foreground classes range between [1, N_CLASS].

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

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
	parser.add_argument('-o', '--output-file', type=str, default=None, help='Path to file in which to save best model.')
	parser.add_argument('-n', '--epochs', type=int, default=5, help='Number of training epochs.')
	parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size.')
	parser.add_argument('-d', '--use-densenet', action="store_true", help='Use DenseNet121 architecture for patch classification.')
	parser.add_argument('-k', '--classes', type=int, default=2, help='Number of classes.')
	parser.add_argument('-c', '--gradient-checkpoints', type=int, default=0, help='Number of gradient checkpoints.')
	parser.add_argument('-A', '--augment-data', action="store_true", help="Use data augmentation (rotation + reflection).")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	xf = None
	
	if args.use_densenet:
		xf = densenet_preprocess()
	
	if args.augment_data:
		data_aug = transforms.Compose([
			transforms.RandomRotation(30),
			transforms.RandomVerticalFlip(),
			transforms.RandomHorizontalFlip()
		])

		if xf is not None:
			xf = transforms.Compose([xf, transforms.ToPILImage(), data_aug])
		else:
			xf = data_aug

	ds = PatchDataset(args.imgdir, args.lbldir, xf)
	n_test = int(0.2 * len(ds))
	trainset, testset = random_split(ds, [len(ds)-n_test, n_test])

	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
	test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
	dataloaders = {"train": train_loader, "val": test_loader}

	x,_ = trainset[0]
	c, h_patch, w_patch = x.shape
	n_classes = args.classes

	if args.use_densenet:
		model = densenet121(n_classes, pretrained=False, checkpoints=args.gradient_checkpoints)
	else:
		model = patchcnn_simple(h_patch, w_patch, c, n_classes, checkpoints=args.gradient_checkpoints)

	loss = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	best_model, train_hist = train_model(model, dataloaders, loss, optimizer, args.epochs, args.output_file)











