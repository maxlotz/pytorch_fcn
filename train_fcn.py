from DatasetLoaders import VOCClassSegBase
from Models import FCN16s
from Utils import make_graph
from Metrics import IOU, seg_Accuracy, Conf_Mat

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


fcn16_torch_path = '/BigDrive/maxlotz/pytorch_models/fcn16s_from_caffe.pth' 
VOC_root = '/BigDrive/maxlotz/datasets/'
save_path = '/BigDrive/maxlotz/pytorch_models/'

model_name = 'VOC_TEST'
log_root = '/home/maxlotz/Thesis/Logs_pytorch/'
graph_path = '/home/maxlotz/Thesis/Figs_pytorch/'

VOC_CLASSES = ['background',
'aeroplane', 'bicycle', 'bird', 'boat',
'bottle', 'bus', 'car', 'cat', 'chair',
'cow', 'diningtable', 'dog', 'horse',
'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor']

train_iters = 20
train_batches = 1
val_batches = 1
disp_batches = 10 # number of batches to display loss after

momentum = 0.99
lr = 1.0e-12
weight_decay = 0.0005

use_gpu = torch.cuda.is_available()
#use_gpu = False

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
    print "Training"
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
            running_loss = 0.0
            print '[{}, {}] loss: {:.4f}'.format(epoch + 1, batch_idx + 1, avg_loss)
            with open(log_root + model_name + '.csv', 'a') as f:
                f.write('{},{},{},{:.4f},{},{}\n'.format(0,epoch, batch_idx + 1, avg_loss, 0, 0))
            running_loss = 0.0
    print "\n"

def test(epoch):
    print "Testing"
    model.eval()
    running_loss = 0.0
    batch_IOU, batch_acc = IOU(), seg_Accuracy()
    epoch_IOU, epoch_acc = IOU(), seg_Accuracy()
    if epoch == train_iters:
        confusion = Conf_Mat(classes=VOC_CLASSES)
    for batch_idx, (data, target) in enumerate(val_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        log_p = F.log_softmax(output, dim=1)
        loss = criterion(log_p, target)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        running_loss += loss.data[0]
        batch_IOU.add(pred, target)
        batch_acc.add(pred, target)
        epoch_IOU.add(pred, target)
        epoch_acc.add(pred, target)
        if epoch == train_iters:
            confusion.add(pred, target)
        if batch_idx % disp_batches == disp_batches-1:
            batch_mean_iou = batch_IOU.get_mean_iou()
            batch_acc = batch_acc.get_accuracy()
            avg_loss = running_loss / disp_batches
            running_loss = 0.0
            print '[{}, {}] loss: {:.4f}\t IOU: {:.4f}\t Acc: {:.4f}'.format(epoch + 1, batch_idx + 1, avg_loss, batch_mean_iou, batch_acc)
            batch_IOU, batch_acc = IOU(), seg_Accuracy()
    epoch_mean_iou = epoch_IOU.get_mean_iou()
    epoch_acc = epoch_acc.get_accuracy()
    with open(log_root + model_name + '.csv', 'a') as f:
        f.write('{},{},{},{:.4f},{:.4f},{:.4f}\n'.format(1, epoch, 0, avg_loss, epoch_acc, epoch_mean_iou))
    if epoch == train_iters:
        confusion.save(VOC_CLASSES, model_name)
        make_graph(log_root + model_name, graph_path + model_name)
    print "\n"

with open(log_root + model_name + '.csv', 'w') as f:
    f.write('set,epoch,batch,loss,accuracy,iou\n')

test(0)
for epoch in xrange(train_iters):
    train(epoch)
    test(epoch+1)