import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

class ImgLabelPair(Dataset):
    """Face Landmarks dataset."""

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

def get_meanstd(dataset):
    # Make sure dataset is transformed to tensor first
    # ONLY USE ON SMALL DATASET <100,000 imgs, TAKES LONG TIME
    ln = len(dataset)
    sz = dataset[0][0].size()
    print "Loading dataset into tensor"
    print "-"*10
    tensor = torch.FloatTensor(ln, sz[0], sz[1], sz[2]).zero_()
    for idx, data in enumerate(dataset):
        if (idx % 100) == 0:
            print "loading image {} of {}".format(idx, ln)
        tensor[idx,:,:,:] = data[0]
    print "-"*10
    print "Dataset loaded into tensor"
    R, G, B = tensor[:,0,:,:], tensor[:,1,:,:], tensor[:,2,:,:]
    mean = [R.mean(), G.mean(), B.mean()]
    std = [R.std(), G.std(), B.std()]
    return mean, std