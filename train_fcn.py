from DatasetLoaders import VOCClassSegBase
from Models import FCN16s
from Utils import make_graph

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch.autograd import Variable

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc


vgg16_npy_path = '/BigDrive/maxlotz/pytorch_models/vgg16.npy'
fcn16_torch_path = '/BigDrive/maxlotz/pytorch_models/fcn16s_from_caffe.pth' 
VOC_root = '/BigDrive/maxlotz/datasets/'
save_path = '/BigDrive/maxlotz/pytorch_models/'
log_path = '/home/maxlotz/Thesis/Logs_pytorch/fcn16_rgb.csv'
graph_path = '/home/maxlotz/Thesis/Figs_pytorch/fcn16_rgb.png'

testimg = '/BigDrive/maxlotz/datasets/VOCPascal/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg'
testlbl = '/BigDrive/maxlotz/datasets/VOCPascal/VOCdevkit/VOC2012/SegmentationClass/2007_000033.png'

train_batches = 1
val_batches = 1
disp_batches = 10 # number of batches to display loss after

use_gpu = torch.cuda.is_available()
use_gpu = False

# Create train and test dataloaders
train_set = VOCClassSegBase(VOC_root, 'train', True)
train_loader = DataLoader(train_set, batch_size=train_batches,
                                          shuffle=True, num_workers=4)
val_set = VOCClassSegBase(VOC_root, 'val', True)
val_loader = DataLoader(val_set, batch_size=val_batches,
                                          shuffle=False, num_workers=4)

model = FCN16s()
model.load_state_dict(torch.load(fcn16_torch_path))

if use_gpu:
	model.cuda()
a = iter(val_loader)
data, target = a.next()

if use_gpu:
    data, target = data.cuda(), target.cuda()
data, target = Variable(data), Variable(target)
out = model(data)

criterion = nn.NLLLoss(ignore_index=255)
log_p = F.log_softmax(out, dim=1)
loss = criterion(log_p, target) # gives same result as Kentaro's manual method
print loss.data[0]



'''
vgg16_weight_dict = np.load(vgg16_npy_path, encoding='latin1').item()

for k, v in vgg16_weight_dict.items():
	if 'conv' in k:
		resized = np.moveaxis(v[0],(0,1,2,3),(2,3,1,0))
		print '{}\t{}'.format(k, resized.shape)
	elif 'fc6' in k:
		resized = np.moveaxis(np.reshape(v[0],(7,7,512,4096)),(0,1,2,3),(2,3,1,0))
		print '{}\t{}'.format(k, resized.shape)
	elif 'fc7' in k:
		resized = np.moveaxis(np.reshape(v[0],(1,1,4096,4096)),(0,1,2,3),(2,3,1,0))
		print '{}\t{}'.format(k, resized.shape)
	else:
		print '{}\t{}'.format(k, v[0].shape)

for name, param in model.named_parameters():
    print '{}\t{}'.format(name, param.size())
'''
