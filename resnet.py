# A simplified ResNet for patch classification in order to test end-to-end training procedure.

import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# Simplified ResNet without Batch Normalization layers and an optional thinning argument
# to reduce the number of filters in each layer.
class ResNetSimple(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 thin=1):
        super(ResNetSimple, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64//thin, layers[0])
        self.layer2 = self._make_layer(block, 128//thin, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256//thin, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512//thin, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512//thin * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # Returns gradient norm for each layer of the model.
    def gradient_norms(self):
        all_norms = []
        for l in [self.conv1, self.layer1, self.layer2, self.layer3, self.layer4, self.fc]:
            layer_norm = 0
            for p in l.parameters():
                if p.grad is not None:
                    pnorm = p.grad.data.norm(2)
                    layer_norm += pnorm.item()**2
            layer_norm = layer_norm**(0.5)
            all_norms.append(layer_norm)
        return all_norms


class ResNetSeg(ResNetSimple):
    
    def __init__(self, block, layers, out_dims, num_classes=1000,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 thin=1):
        super(ResNetSeg, self).__init__(block, layers, num_classes, groups, 
            width_per_group, replace_stride_with_dilation, thin)

        self.avgpool = nn.AdaptiveAvgPool2d(out_dims)
        self.conv2 = nn.Conv2d(512//thin * block.expansion, num_classes, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.conv2(x)

        return x


def resnet18(n_classes, thin=1):
    return ResNetSimple(BasicBlock, [2, 2, 2, 2], num_classes=n_classes, thin=thin)

def resnetseg34(n_classes, out_dims, thin=1):
    return ResNetSeg(BasicBlock, [3,4,6,3], out_dims, num_classes=n_classes, thin=thin)

######################################

import time, copy
from utils import class_auroc


def train_rnseg(model, dataloaders, criterion, optimizer, num_epochs, outfile=None):
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
            running_totals = 0

            epoch_labels, epoch_softmax = [],[]

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

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

                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    epoch_labels.append(labels)
                    epoch_softmax.append(nn.functional.softmax(outputs, dim=1))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) 
                running_corrects += torch.sum(preds == labels.data)
                running_totals += len(labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

            epoch_labels = np.concatenate([x.cpu().data.numpy() for x in epoch_labels])
            epoch_softmax = np.concatenate([x.cpu().data.numpy() for x in epoch_softmax])
            auroc = class_auroc(epoch_softmax, epoch_labels)
            print('{} AUROC: {}'.format(phase, "\t".join(list(map(str, auroc)))))

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

######################################

import os, sys
import numpy as np
sys.path.append("..")
from datasets import PatchDataset, StitchGridDataset
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt 

if __name__ == "__main__":
    pd = PatchDataset(os.path.expanduser("~/Desktop/aba_stdataset_20200212/imgs128"),
        os.path.expanduser("~/Desktop/aba_stdataset_20200212/lbls128"))
    dl = DataLoader(pd, batch_size=10, shuffle=True)

    rnet = resnet18(13, thin=4)

    x,y = next(iter(dl))
    print(rnet(x))
    print(y)

    img_dir = os.path.expanduser("~/Desktop/aba_stdataset_20200212/imgs256/")
    lbl_dir = os.path.expanduser("~/Desktop/aba_stdataset_20200212/lbls256/")
    sd = StitchGridDataset(img_dir, lbl_dir)
    dl = DataLoader(sd, batch_size=1)

    x2, y2 = next(iter(dl))
    print(x2.shape)

    fig = plt.figure()
    x2_arr = x2.data.numpy()[0]
    x2_arr = np.moveaxis(x2_arr, 0, 2)
    plt.imshow(x2_arr)
    plt.show()

    rnseg = resnetseg34(13, (32,49), thin=4)
    out = rnseg(x2)
    print(out.shape)

