import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import weight_norm
from torch.autograd import Variable




class CNN(nn.Module):
    def __init__(self, p=0.5, channels=3, num_class=100):
        super(CNN, self).__init__()
        self.conv1a = weight_norm(nn.Conv2d(channels, 128, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1c = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2) 
        self.drop1 = nn.Dropout(p)
        self.conv2a = weight_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn2c = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(p)
        self.conv3a = weight_norm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=False))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False))
        self.bn3c = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(6)
        self.linear = nn.Linear(128, num_class)

        self._init_weights()

    def _init_weights(self):
        """Initialize the weights."""
        self.conv1a.weight.data = init.kaiming_uniform_(self.conv1a.weight.data)
        self.conv1b.weight.data = init.kaiming_uniform_(self.conv1b.weight.data)
        self.conv1c.weight.data = init.kaiming_uniform_(self.conv1c.weight.data)

        self.conv2a.weight.data = init.kaiming_uniform_(self.conv2a.weight.data)
        self.conv2b.weight.data = init.kaiming_uniform_(self.conv2b.weight.data)
        self.conv2c.weight.data = init.kaiming_uniform_(self.conv2c.weight.data)
        
        self.conv3a.weight.data = init.kaiming_uniform_(self.conv3a.weight.data)
        self.conv3b.weight.data = init.kaiming_uniform_(self.conv3b.weight.data)
        self.conv3c.weight.data = init.kaiming_uniform_(self.conv3c.weight.data)
        self.linear.weight.data = init.kaiming_uniform_(self.linear.weight.data)


    def forward(self, x):
        out = F.leaky_relu(self.bn1a(self.conv1a(x)), negative_slope=0.1)
        out = F.leaky_relu(self.bn1b(self.conv1b(out)), negative_slope=0.1)
        out = F.leaky_relu(self.bn1c(self.conv1c(out)), negative_slope=0.1)
        out = self.pool1(out)
        out = self.drop1(out)

        out = F.leaky_relu(self.bn2a(self.conv2a(out)), negative_slope=0.1)
        out = F.leaky_relu(self.bn2b(self.conv2b(out)), negative_slope=0.1)
        out = F.leaky_relu(self.bn2c(self.conv2c(out)), negative_slope=0.1)
        out = self.pool2(out)
        out = self.drop2(out)

        out = F.leaky_relu(self.bn3a(self.conv3a(out)), negative_slope=0.1)
        out = F.leaky_relu(self.bn3b(self.conv3b(out)), negative_slope=0.1)
        out = F.leaky_relu(self.bn3c(self.conv3c(out)), negative_slope=0.1)
        out = self.pool3(out)

        out = out.view(out.size(0),-1)
        y = self.linear(out)

        return y, out




