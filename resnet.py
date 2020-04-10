import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class AttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super(AttentionLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_1, b_1, flag):
        if flag==1:
            out_c = F.linear(features, W_1, b_1)
            # out = F.linear(out_1, W_2, b_2)
            out = out_c - out_c.max()
            out = out.exp()
            out = out.sum(1, keepdim=True)
            alpha = out / out.sum(0)
        else:
            alpha = torch.zeros(features.size(0),1)
                
        return torch.squeeze(alpha) *features.size(0)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 =nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)    
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)    
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet_Part(nn.Module):
    def __init__(self, block, num_blocks, class_num=10):
        super(ResNet_Part, self).__init__()
        self.in_planes=64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AvgPool2d(4)
        # self.att_layer = AttentionLayer(512*block.expansion)
        self.linear = nn.Linear(512,class_num)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides =[stride]+[1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes *block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out =self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)
        y = self.linear(out)

        return y, out

        
    
def ResNet18(class_num=10):
    return ResNet_Part(BasicBlock, [2,2,2,2], class_num=class_num)

def ResNet34():
    return ResNet_Part(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet_Part(Bottleneck, [3,4,6, 3])

def ResNet101():
    return ResNet_Part(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet_Part(Bottleneck, [3,8,36,3])

