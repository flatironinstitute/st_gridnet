import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

from torchvision import models, transforms

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
			out = checkpoint_sequential(self.cnn, self.checkpoints, x)

		out = out.reshape(out.size(0), -1)
		out = self.fcn(out)

		return out

def patchcnn_simple(h_patch, w_patch, c, n_classes, checkpoints):
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
def densenet121(n_classes, pretrained=False):
	dnet = models.densenet121(pretrained)
	dnet.classifier = nn.Linear(1024, n_classes)
	return dnet

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
