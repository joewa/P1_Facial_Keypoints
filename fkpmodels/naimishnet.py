## DONE: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


# Inspired by NaimishNet https://arxiv.org/pdf/1710.00977.pdf
class YaNaimishNet1(nn.Module):

    def __init__(self):
        super(YaNaimishNet1, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (128-5)/1 +1 = 124
        # the output Tensor for one image, will have the dimensions: (32, 124, 124)
        # after one pool layer, this becomes (32, 62, 62)
        self.conv1 = nn.Conv2d(3, 32, 5)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (62-3)/1 +1 = 60
        # the output tensor will have dimensions: (64, 60, 60)
        # after another pool layer this becomes (64, 30, 30)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # 3rd conv layer: 64 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (30-3)/1 +1 = 28
        # the output tensor will have dimensions: (128, 28, 28)
        # after another pool layer this becomes (128, 14, 14)
        self.conv3 = nn.Conv2d(64, 128, 3)

        # 4th conv layer: 128 inputs, 256 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (14-3)/1 +1 = 12
        # the output tensor will have dimensions: (128, 12, 12)
        # after another pool layer this becomes (256, 6, 6)
        self.conv4 = nn.Conv2d(128, 256, 3)
        # 5th conv layer: 256 inputs, 512 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (6-1)/1 +1 = 6
        # the output tensor will have dimensions: (512, 10, 10)
        # after another pool layer this becomes (512, 3, 3)
        self.conv5 = nn.Conv2d(256, 512, 1)

        # 256 outputs * the 12*12 filtered/pooled map size
        # self.fc1 = nn.Linear(256*12*12, 4608)
        # self.fc2 = nn.Linear(4608, 2304)
        # self.fc3 = nn.Linear(2304, 2304)
        # self.fc4 = nn.Linear(2304, 68*2)
        # 256 outputs * the 12*12 filtered/pooled map size
        self.fc1 = nn.Linear(512*3*3, 4608)
        self.fc2 = nn.Linear(4608, 2304)
        self.fc3 = nn.Linear(2304, 2304)
        self.fc4 = nn.Linear(2304, 16*2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # four conv/relu + pool layers
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.pool(F.elu(self.conv4(x)))
        x = self.pool(F.elu(self.conv5(x)))

        # three linear layers with dropout in between
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        #x = self.fc1_drop(x)
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x


class YaNaimishNet2(nn.Module):

    def __init__(self):
        super(YaNaimishNet2, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (128-5)/1 +1 = 124
        # the output Tensor for one image, will have the dimensions: (64, 124, 124)
        # after one pool layer, this becomes (64, 62, 62)
        self.conv1 = nn.Conv2d(3, 64, 5)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (62-3)/1 +1 = 60
        # the output tensor will have dimensions: (128, 60, 60)
        # after another pool layer this becomes (128, 30, 30)
        self.conv2 = nn.Conv2d(64, 128, 3)

        # 3rd conv layer: 64 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (30-3)/1 +1 = 28
        # the output tensor will have dimensions: (256, 28, 28)
        # after another pool layer this becomes (256, 14, 14)
        self.conv3 = nn.Conv2d(128, 256, 3)

        # 4th conv layer: 128 inputs, 256 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (14-3)/1 +1 = 12
        # the output tensor will have dimensions: (512, 12, 12)
        # after another pool layer this becomes (512, 6, 6)
        self.conv4 = nn.Conv2d(256, 512, 3)
        # 5th conv layer: 256 inputs, 512 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (6-1)/1 +1 = 6
        # the output tensor will have dimensions: (1024, 6, 6)
        # after another pool layer this becomes (1024, 3, 3)
        self.conv5 = nn.Conv2d(512, 1024, 1)

        # 256 outputs * the 12*12 filtered/pooled map size
        # self.fc1 = nn.Linear(256*12*12, 4608)
        # self.fc2 = nn.Linear(4608, 2304)
        # self.fc3 = nn.Linear(2304, 2304)
        # self.fc4 = nn.Linear(2304, 68*2)
        # 256 outputs * the 12*12 filtered/pooled map size
        self.fc1 = nn.Linear(1024*3*3, 4608)
        self.fc2 = nn.Linear(4608, 4608)
        self.fc3 = nn.Linear(4608, 16*2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # four conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # three linear layers with dropout in between
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        #x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x


class YaNaimishNet3(nn.Module):

    def __init__(self):
        super(YaNaimishNet3, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-9)/1 +1 = 216
        # the output Tensor for one image, will have the dimensions: (32, 216, 216)
        # after one pool layer, this becomes (64, 108, 108)
        self.conv1 = nn.Conv2d(3, 32, 9)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool_small = nn.MaxPool2d(2, 2)
        self.pool_big = nn.MaxPool2d(4, 2)

        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output tensor will have dimensions: (128, 108, 108)
        # after another pool layer this becomes (128, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 5)

        # 3rd conv layer: 64 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output tensor will have dimensions: (256, 52, 52)
        # after another pool layer this becomes (256, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 3)

        # 4th conv layer: 128 inputs, 256 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # the output tensor will have dimensions: (256, 24, 24)
        # after another pool layer this becomes (256, 12, 12)
        self.conv4 = nn.Conv2d(128, 256, 1)

        # 256 outputs * the 12*12 filtered/pooled map size
        # self.fc1 = nn.Linear(256*12*12, 4608)
        # self.fc2 = nn.Linear(4608, 2304)
        # self.fc3 = nn.Linear(2304, 2304)
        # self.fc4 = nn.Linear(2304, 68*2)
        # 256 outputs * the 12*12 filtered/pooled map size
        self.fc1 = nn.Linear(6400, 3200)
        self.fc2 = nn.Linear(3200, 1600)
        self.fc3 = nn.Linear(1600, 16*2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # four conv/relu + pool layers
        x = self.pool_big(F.relu(self.conv1(x)))
        x = self.pool_big(F.relu(self.conv2(x)))
        x = self.pool_small(F.relu(self.conv3(x)))
        x = self.pool_big(F.relu(self.conv4(x)))
        # three linear layers with dropout in between
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        #x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
