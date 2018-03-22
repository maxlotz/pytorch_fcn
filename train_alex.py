from DatasetLoaders import Img_LabelInt_Pair
from Models import AlexNet
from Utils import make_graph
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch.autograd import Variable

weights_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
train_file = '/home/maxlotz/Thesis/file_lists/rgb_train_1.txt'
test_file = '/home/maxlotz/Thesis/file_lists/rgb_test_1.txt'
class_file = '/home/maxlotz/Thesis/file_lists/classes.txt'
save_path = '/BigDrive/maxlotz/pytorch_models/'
log_path = '/home/maxlotz/Thesis/Logs_pytorch/VOC_TEST.csv'
graph_path = '/home/maxlotz/Thesis/Figs_pytorch/Alexnet_rgb.png'

train_batches = 64
test_batches = 64
disp_batches = 10 # number of batches to display loss after

rgb_mean = [0.14735279922924333, 0.131836718919208, 0.11958479305748611]
rgb_std = [0.2428308948828459, 0.22527291067069058, 0.22105494379177684]

use_gpu = torch.cuda.is_available()

# Create transforms, crop, convert to torch tensor then normalise
transform = transforms.Compose(
    [transforms.RandomResizedCrop(227),
     transforms.ToTensor(),
     transforms.Normalize(rgb_mean, rgb_std)])

# Create train and test dataloaders
train_set = Img_LabelInt_Pair(train_file, transform)
train_loader = DataLoader(train_set, batch_size=train_batches,
                                          shuffle=True, num_workers=4)
test_set = Img_LabelInt_Pair(test_file, transform)
test_loader = DataLoader(test_set, batch_size=test_batches,
                                          shuffle=True, num_workers=4)

# Initialise model and copy over pretrained weights with same size
model = AlexNet(num_classes=51)
model_dict = model.state_dict()
weights_dict = model_zoo.load_url(weights_url)
new_dict = {k: v for k, v in weights_dict.items() if k not in ['classifier.6.weight', 'classifier.6.bias']}
model_dict.update(new_dict)
model.load_state_dict(model_dict)

# Train model and print training loss
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if batch_idx % disp_batches == disp_batches-1:
            avg_loss = running_loss / disp_batches
            print '[{}, {}] loss: {:.4f}'.format(epoch + 1, batch_idx + 1, avg_loss)
            with open(log_path, 'a') as f:
                f.write('{},{},{:.4f},{},{}\n'.format(epoch, batch_idx + 1, avg_loss, 0, 0))
            running_loss = 0.0

# Test model and print test loss and accuracy
def test(epoch):
    model.eval()
    test_loss, correct = 0, 0
    for data, target in test_loader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader)
    accuracy = correct*100.0/len(test_loader.dataset)
    print 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss.data[0], correct, len(test_loader.dataset), accuracy)
    with open(log_path, 'a') as f:
        f.write('{},{},{:.4f},{:.4f},{}\n'.format(epoch, 0, test_loss.data[0], accuracy, 1))

if use_gpu:
	model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

with open(log_path, 'w') as f:
    f.write('epoch,batch,loss,accuracy,set\n')

'''
for epoch in range(2):
    train(epoch)
    test(epoch+1)
    model_dict = model.state_dict()
    torch.save(model_dict, save_path + 'Alexnet_rgb_' + str(epoch) + '.pth')
'''

