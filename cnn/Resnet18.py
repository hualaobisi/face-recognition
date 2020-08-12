# ResNet model refers to torchvision official model https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# Dropblock refers to https://github.com/miguelvr/dropblock
'''Prepare Olivetti Dataset'''
import torch
from torchvision import datasets,transforms

# the standard input for resnet18 is 224*224, 3 channels
# In the transform part reshape the input images

trainset = datasets.ImageFolder(
    root='./data/train',
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.186,0.186,0.186)),

    ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)

testset = datasets.ImageFolder(root='./data/test',
        transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.47,0.47,0.47), (0.201,0.201,0.201)),
    ]))

testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)


'''ResNet-18 Model'''
# This is the Pytorch official Resnet class
# Add in dropblock to control the overfitting

import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D, LinearScheduler

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=40, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=0.1, block_size=7),
            start_value=0.,
            stop_value=0.1,
            nr_steps=500,
        ) # The output for last layer is 7 * 7, the drop block size should not exceed the whole channel size.

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self,x):
        self.dropblock.step()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropblock(x)
        x = self.layer4(x)
        x = self.dropblock(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, inplanes, planes, pretrained, progress, **kwargs):
    model = ResNet(inplanes, planes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


'''Trainning the Model'''

from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import time

# Run on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Olivetti Training')
parser.add_argument('--outf', default='./model', help='folder to output images and model checkpoints')
parser.add_argument('--net', default='./model', help="path to net (to continue training)")
args = parser.parse_args()


EPOCH = 200
pre_epoch = 0
LR = 0.01


net = resnet18().to(device)

# SGD optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


if __name__ == "__main__":
    best_acc = 0
    print("Start Training, Resnet-18!")
    with open("acc_resnet18_dropblock.txt", "w") as f:
        with open("log_resnet18_dropblock.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):

                if epoch >= 100:
                    optimizer.param_groups[0]['lr'] = LR * 0.1
                if epoch >= 150:
                    optimizer.param_groups[0]['lr'] = LR * 0.1 * 0.1

                print('\nEpoch: %d' % (epoch + 1))
                print('learning rate = ' + str(optimizer.param_groups[0]['lr']))
                start = time.time()
                net.train() # just do dropblock at training process
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
                        outputs = net.forward(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()

                    end = time.time()
                    total_time = end-start
                    print('Training Accuracy：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    print('Training Time: %.3f'% total_time)
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/Resnet18_dropblock_4_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%,Learning rate= %s,Time= %.3f" % (epoch + 1, acc,optimizer.param_groups[0]['lr'],total_time))
                    f.write('\n')
                    f.flush()
                    if acc > best_acc:
                        f3 = open("best_acc_resnet18_dropblock.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.write('\n')
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)