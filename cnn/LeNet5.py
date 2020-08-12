# LeNet5 model refers to https://github.com/zaoyifan/LeNet-5-for-Face-Recognition-with-ORL/blob/master/lenet.py

'''Prepare Olivetti Dataset'''
import torch
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torchvision import datasets,transforms


# Reshape the input size to be 32 * 32
# The input channel = 1, just the gray.
# In the transform part reshape the input images

trainset = datasets.ImageFolder(
    root='./data/train',
    transform=transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.452], [0.197]),
    ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

testset = datasets.ImageFolder(root='./data/test',
                               transform=transforms.Compose([
                                   transforms.Grayscale(1),
                                   transforms.Resize(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.452], [0.204]),
                               ]))

testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)

'''LeNet5 Model'''
# This model refers to the  Y.LeCun's model for Handwritten digits recognition

from torch import optim
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1, 6, 5)
       self.bn1 = nn.BatchNorm2d(6)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.bn2 = nn.BatchNorm2d(16)
       self.conv3 = nn.Conv2d(16,120,5)
       self.bn3 = nn.BatchNorm2d(120)
       self.fc1 = nn.Linear(1920,84)
       self.bn4 = nn.BatchNorm1d(84)
       self.fc2 = nn.Linear(84, 40)

   def forward(self, x):
       x = self.conv1(x)
       x = self.bn1(x)
       x = self.conv2(x)
       x = F.max_pool2d(x, (2, 2))
       x = self.bn2(x)
       x = self.conv3(x)
       x = F.max_pool2d(x, (2, 2))
       x = self.bn3(x)
       x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
       x = F.relu(self.fc1(x))
       x = self.bn4(x)
       x = F.relu(self.fc2(x))
       return x


'''Trainning the Model'''

from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import time

# run on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Olivetti Training')
parser.add_argument('--outf', default='./model', help='folder to output images and model checkpoints')
parser.add_argument('--net', default='./model', help="path to net (to continue training)")
args = parser.parse_args()


EPOCH = 200
pre_epoch = 0
batch_size = 10

net = LeNet5().to(device)

criterion = nn.CrossEntropyLoss()

# import a scheduled optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


if __name__ == "__main__":
    best_acc = 0
    print("Start Training, LeNet5!")
    with open("acc_lenet_BN.txt", "w") as f:
        with open("log_lenet_BN.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):

                # if epoch >= 50:
                #     optimizer.param_groups[0]['lr'] = LR * 0.1
                # if epoch >= 100:
                #     optimizer.param_groups[0]['lr'] = LR * 0.1 * 0.1
                # if epoch >= 150:
                #     optimizer.param_groups[0]['lr'] = LR * 0.1 * 0.1 * 0.1

                print('\nEpoch: %d' % (epoch + 1))
                print('learning rate = ' + str(optimizer.param_groups[0]['lr']))
                start = time.time()
                net.train()
                scheduler.step()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0

                for i, data in enumerate(trainloader, 0):
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = net.forward(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.06f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.06f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()


                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for i, data in enumerate(testloader, 0):
                        net.eval()
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = net(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()

                    end = time.time()
                    total_time = end-start
                    print('Training Accuracy：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    print('Training Time: %.3f'% total_time)
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/LeNet5_BN_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%,Learning rate= %s,Time= %.3f" % (epoch + 1, acc,optimizer.param_groups[0]['lr'],total_time))
                    f.write('\n')
                    f.flush()
                    if acc > best_acc:
                        f3 = open("best_acc_lenet_BN.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.write('\n')
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)