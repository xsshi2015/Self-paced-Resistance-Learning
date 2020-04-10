import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import weight_norm
from torch.autograd import Variable
# from utils import GaussianNoise, savetime, save_exp

import weight_loss as WL




class CNN(nn.Module):
    def __init__(self, p=0.5, num_class=100):
        super(CNN, self).__init__()
        # self.gn = GaussianNoise(batch_size, input_shape=(1, 32, 32), std=std)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False))
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
        # if self.training:
        #     x = self.gn(x)
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


# class Custom_Loss(torch.nn.Module):
#     def __init__(self, num_bits=128):
#         super(Custom_Loss, self).__init__()
#         self.num_bits = num_bits

#     def forward(self, out, out_f, target, target_f, u_target, u_target_f, u_p, u_p_f, labels, mask_flag, u_w, u_w_m, epoch):
        
#         def labeled_mse_loss(out1, out2, epoch):
#             if epoch>0:
#                 cond = (mask_flag > 0)
#                 nnz = torch.nonzero(cond)
#                 nbsup = len(nnz)
#                 if nbsup>0:
#                     quad_diff = torch.sum((out1[cond,:]-out2)**2) / out2.data.nelement() 
#                 else:
#                     quad_diff = 0.0
#                 return quad_diff
#             else:
#                 return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False)

#         def unlabeled_mse_loss(out1, u_out2, u_p, epoch):
#             if epoch>0:
#                 cond_1= (mask_flag == 0)
#                 nnz_1 = torch.nonzero(cond_1)
#                 nbsup_1 = len(nnz_1)
#                 if nbsup_1>0:
#                     quad_diff = torch.sum((torch.unsqueeze(u_p, dim=1).expand_as(u_out2)*(out1[cond_1,:].repeat(1, u_out2.size(0)//nbsup_1).view(u_out2.size(0), u_out2.size(1)) - u_out2))**2) / u_out2.data.nelement() 
#                 else:
#                     quad_diff = 0.0

#                 return quad_diff 
#             else:
#                 return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False)
                

            
#         def masked_crossentropy(out, labels, mask_flag):
#             cond = (mask_flag > 0)
#             nnz = torch.nonzero(cond)
#             nbsup = len(nnz)
#             # check if labeled samples in batch, return 0 if none
#             if nbsup > 0:
#                 masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
#                 masked_labels = labels[cond]
#                 loss =  F.cross_entropy(masked_outputs, masked_labels)
#                 return loss, nbsup

#             # if nbsup > 0:
#             #     masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
#             #     masked_labels = labels[cond,:]
#             #     loss =  F.binary_cross_entropy_with_logits(masked_outputs, masked_labels)
#             #     return loss, nbsup
                
#             return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0

#         sup_loss, nbsup = masked_crossentropy(out, labels, mask_flag)
#         m_loss_1 = labeled_mse_loss(out, target,  epoch)
#         m_loss_2 = labeled_mse_loss(out_f, target_f,  epoch)
        
#         u_m_loss_1 = unlabeled_mse_loss(out,  u_target, u_p, epoch)
#         u_m_loss_2 = unlabeled_mse_loss(out_f,  u_target_f, u_p_f, epoch)

#         return sup_loss + u_w*m_loss_1 + u_w*m_loss_2 + 0.1*u_w*u_m_loss_1 + 0.1*u_w*u_m_loss_2



