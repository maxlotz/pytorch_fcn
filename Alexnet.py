from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from DatasetLoaders import ImgLabelPair

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

filedir = "/home/maxlotz/Thesis/file_lists"
trainfile, testfile, classfile = '/rgb_train_1.txt' , '/rgb_test_1.txt', '/classes.txt'

trainset = ImgLabelPair(filedir + trainfile, data_transform)
trainloader = DataLoader(trainset, batch_size=4,
                        shuffle=True, num_workers=4)
trainset = ImgLabelPair(filedir + testfile, data_transform)
trainloader = DataLoader(trainset, batch_size=4,
                        shuffle=True, num_workers=4)

with open(filedir + classfile, 'r') as f:
    classes = f.read().splitlines()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(utils.make_grid(images))
plt.show()
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))