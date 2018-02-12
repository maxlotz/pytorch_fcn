from DatasetLoaders import ImgLabelPair
from Models import AlexNet
import torch
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from DatasetLoaders import get_meanstd

weights_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
train_file = '/home/maxlotz/Thesis/file_lists/rgb_train_1.txt'
test_file = '/home/maxlotz/Thesis/file_lists/rgb_test_1.txt'

train_batches = 256
test_batches = 64

rgb_mean = [0.14735279922924333, 0.131836718919208, 0.11958479305748611]
rgb_std = [0.2428308948828459, 0.22527291067069058, 0.22105494379177684]

# Initialise model and copy over pretrained weights with same size
model = AlexNet(num_classes=51)
model_dict = model.state_dict()
weights_dict = model_zoo.load_url(weights_url)
new_dict = {k: v for k, v in weights_dict.items() if k not in ['classifier.6.weight', 'classifier.6.bias']}
model_dict.update(new_dict)
model.load_state_dict(model_dict)

# Create transforms, subtract mean and normalise
transform = transforms.Compose(
    [transforms.RandomSizedCrop(227),
     transforms.ToTensor(),
     transforms.Normalize(rgb_mean, rgb_std)])

#create train and test dataloaders
train_set = ImgLabelPair(train_file, transform)
train_loader = DataLoader(train_set, batch_size=train_batches,
                                          shuffle=True, num_workers=4)
test_set = ImgLabelPair(test_file, transform)
test_loader = DataLoader(test_set, batch_size=test_batches,
                                          shuffle=True, num_workers=4)


