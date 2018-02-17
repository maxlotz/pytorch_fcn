from DatasetLoaders import VOCClassSegBase
from Models import FCN16s
from Utils import make_graph
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

weights_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
VOC_root = '/BigDrive/maxlotz/datasets/'
save_path = '/BigDrive/maxlotz/pytorch_models/'
log_path = '/home/maxlotz/Thesis/Logs_pytorch/Alexnet_rgb.csv'
graph_path = '/home/maxlotz/Thesis/Figs_pytorch/Alexnet_rgb.png'

testimg = '/BigDrive/maxlotz/datasets/VOCPascal/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg'
testlbl = '/BigDrive/maxlotz/datasets/VOCPascal/VOCdevkit/VOC2012/SegmentationClass/2007_000033.png'

train_batches = 1
val_batches = 1
disp_batches = 10 # number of batches to display loss after

use_gpu = torch.cuda.is_available()

# Create train and test dataloaders
train_set = VOCClassSegBase(VOC_root, 'train', True)
train_loader = DataLoader(train_set, batch_size=train_batches,
                                          shuffle=True, num_workers=4)
val_set = VOCClassSegBase(VOC_root, 'val', True)
val_loader = DataLoader(val_set, batch_size=val_batches,
                                          shuffle=True, num_workers=4)

model = FCN16s()
if use_gpu:
	model.cuda()

a = iter(val_loader)
data, target = a.next()
if use_gpu:
    data, target = data.cuda(), target.cuda()
data, target = Variable(data), Variable(target)

out = model(data) 
print out.shape