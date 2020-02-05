import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from torchvision import models

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

''' Modified DenseNet with optional pretraining on ImageNet.
	
	Pre-trained models expect input to be transformed according to the following:
	- Shape (3, H, W) where H, W > 224
	- Loaded into range [0,1]
	- Normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
'''
class DenseNet121(nn.Module):
	def __init__(self, n_classes, pretrained=True):
		self.dnet = models.densenet121(pretrained)
		self.dnet.classifier = nn.Linear(1024, n_classes)

	def forward(self, x):
		return self.dnet(x)
