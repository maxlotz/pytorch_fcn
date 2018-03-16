from DatasetLoaders import VOCClassSegBase
from Models import FCN16s
from Utils import make_graph
from Metrics import iou

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

train_batches = 2
val_batches = 1
disp_batches = 10 # number of batches to display loss after

momentum = 0.99
lr = 1.0e-6
weight_decay = 0.0005

use_gpu = torch.cuda.is_available()
use_gpu = False

# Create train and test dataloaders
train_set = VOCClassSegBase(VOC_root, 'train', True)
train_loader = DataLoader(train_set, batch_size=train_batches,
                                          shuffle=True, num_workers=4)
val_set = VOCClassSegBase(VOC_root, 'val', True)
val_loader = DataLoader(val_set, batch_size=val_batches,
                                          shuffle=True, num_workers=4)

model = FCN16s()
model.load_state_dict(torch.load(fcn16_torch_path))

if use_gpu:
    model.cuda()

criterion = nn.NLLLoss(ignore_index=255)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        log_p = F.log_softmax(output, dim=1)
        loss = criterion(log_p, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if batch_idx % disp_batches == disp_batches-1:
            avg_loss = running_loss / disp_batches
            print '[{}, {}] loss: {:.4f}'.format(epoch + 1, batch_idx + 1, avg_loss)
            with open(log_path, 'a') as f:
                f.write('{},{},{:.4f},{},{}\n'.format(epoch, batch_idx + 1, avg_loss, 0, 0))
            running_loss = 0.0

def test(epoch, num_classes=21):
    model.eval()
    test_loss, correct = 0, 0
    intersection, union, ious = np.zeros(num_classes, dtype=np.int32), np.zeros(num_classes, dtype=np.int32), np.zeros(num_classes)
    for data, target in val_loader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        log_p = F.log_softmax(output, dim=1)
        test_loss += criterion(output, target)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        intersection_, union_ = iou(pred, target, 21, 255)
        intersection += intersection_
        union += union_
        print intersection
        print union
    for un, inter, (idx, iou_) in zip(union, intersection, enumerate(ious)):
        if un == 0.:
            ious[idx] = float('nan')
        else:
            ious[idx] = float(inter)/float(max(un, 1))
    notnan = ious[np.logical_not(np.isnan(ious))]
    mean_ious = notnan/len(notnan)
    print mean_ious


test(0)
'''
pred = output.data.max(1)[1]
accuracy = 100.*(pred == target.data).sum()/np.prod(pred.shape)
    print ious
    notnan = ious[np.logical_not(np.isnan(ious))]
    print notnan
'''