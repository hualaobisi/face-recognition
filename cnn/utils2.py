'''
Calculate the means and std for dataloader transform.
This is for LeNet5 model.
Reshape the input size to 32*32.
For this model the input channel = 1, only the gray.
Method refers to https://www.jianshu.com/p/f49127e7843c
'''

import torch
from torchvision import datasets,transforms
import numpy as np

trainset = datasets.ImageFolder(
    root='./data/train',
    transform=transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(32),
        transforms.ToTensor(),
    ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)

testset = datasets.ImageFolder(root='./data/test',
                               transform=transforms.Compose([
                                   transforms.Grayscale(1),
                                   transforms.Resize(32),
                                   transforms.ToTensor(),
                               ]))

testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)

train_mean = []
train_std = []
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs = inputs.numpy()
    batch_mean = np.mean(inputs, axis=(0, 2, 3))
    batch_std = np.std(inputs, axis=(0, 2, 3))

    train_mean.append(batch_mean)
    train_std.append(batch_std)

train_mean = np.array(train_mean).mean(axis=0)
train_std = np.array(train_std).mean(axis=0)

test_mean = []
test_std = []

for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs = inputs.numpy()
    batch_mean = np.mean(inputs, axis=(0,2, 3))
    batch_std = np.std(inputs, axis=(0,2, 3))

    test_mean.append(batch_mean)
    test_std.append(batch_std)

test_mean = np.array(test_mean).mean(axis=0)
test_std = np.array(test_std).mean(axis=0)

print('Train means:',train_mean)
print('Train std:',train_std)
print('Test means:',test_mean)
print('Test std:',test_std)
