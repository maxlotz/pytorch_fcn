import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import numpy as np
import os.path as osp
import scipy.misc as misc


class Img_LabelInt_Pair(Dataset):
    def __init__(self, data_file, transform=None):
        """
        Args:
            data_file (string): Path to .txt file containing "img_file idx" on each line 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(data_file, 'r') as f:
            self.data = f.read().splitlines()

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx].split(' ')
        with Image.open(pair[0]) as img:
            img = img.convert('RGB')
        label = int(pair[1])
        sample = (img, label)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

# taken from github of wkentaro and modified
class VOCClassSegBase(Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', transform=False, subtract_mean=True):
        self.root = root
        self.split = split
        self._transform = transform
        self.resize_size = (256, 256)
        self.subtract_mean = subtract_mean

        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join(self.root, 'VOCPascal/VOCdevkit/VOC2012')
        self.files = {}
        for split in ['train', 'val']:
            self.files[split] = []
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        
        # load image
        img_file = data_file['img']
        img = Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        img = misc.imresize(img, self.resize_size, interp='bilinear')

        # load label
        lbl_file = data_file['lbl']
        lbl = Image.open(lbl_file)
        lbl_ = np.array(lbl)
        # note resize gives type uint8
        lbl = misc.imresize(lbl, self.resize_size, interp='nearest')
        lbl = lbl.astype(np.int32)
        #lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        if self.subtract_mean:
            img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        
        lbl = torch.from_numpy(lbl).long()
        
        return img, lbl


    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl

class NYUV2Seg(VOCClassSegBase):
    class_names = [
        'wall',
        'floor',
        'cabinet',
        'bed',
        'chair',
        'sofa',
        'table',
        'door',
        'window',
        'bookshelf',
        'picture',
        'counter',
        'blinds',
        'desk',
        'shelves',
        'curtain',
        'dresser',
        'pillow',
        'mirror',
        'floor mat',
        'clothes',
        'ceiling',
        'books',
        'refridgerator',
        'television',
        'paper',
        'towel',
        'shower curtain',
        'box',
        'whiteboard',
        'person',
        'night stand',
        'toilet',
        'sink',
        'lamp',
        'bathtub',
        'bag',
        'otherstructure',
        'otherfurniture',
        'otherprop',
    ]

    #mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, imgroot, lblroot, idx, split, transform, subtract_mean):
        self.imgroot = imgroot
        self.lblroot = lblroot
        self.idx = idx
        self.split = split
        self.files = {}
        self._transform = transform
        self.resize_size = (256, 256)
        self.subtract_mean = subtract_mean

        if self.idx > 9:
            print "error"

        self.files['train'], self.files['val'] = [], []
        for i in xrange(1449):
            if (i+1+self.idx) % 10 == 0:
                self.files['val'].append({
                    'img' : self.imgroot + str(i+1) + '.png',
                    'lbl' : self.lblroot + str(i+1) + '.png',
                    })
            else:
                self.files['train'].append({
                    'img' : self.imgroot + str(i+1) + '.png',
                    'lbl' : self.lblroot + str(i+1) + '.png',
                    })