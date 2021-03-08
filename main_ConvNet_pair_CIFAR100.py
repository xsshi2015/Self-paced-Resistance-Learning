from __future__ import print_function
'''

Resources:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#autograd
https://discuss.pytorch.org/t/convert-numpy-to-pytorch-dataset/743
http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#compose-transforms

https://github.com/jcjohnson/pytorch-examples#pytorch-custom-nn-modules
http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#autograd
http://pytorch.org/docs/master/nn.html#convolution-layers
'''

import os
import sys
import time
import shutil
import argparse
import torchvision
import numpy as np
from scipy import io

import torch

import torch.nn as nn
from pprint import pprint
import torch.optim as optim
import torch.nn.functional as F


from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torchvision.models.resnet import model_urls
from torch.utils.data.sampler import SubsetRandomSampler

from utils.stat_utils import AverageMeter
from data.Cifar100Dataset import Cifar100Dataset as Cifar100
import data.data_utils as du
from model.ConvNet import CNN 

from loss.weight_loss import CrossEntropyLoss as CE
from data.data_transform import DataTransform as DT
import data.Generate_noisy_labels as GN


parser = argparse.ArgumentParser(description='PyTorch CoadjutantHashing Training')
parser.add_argument('-d', '--data', metavar='DIR', help='path to dataset (default: ./data)', default='./data')
parser.add_argument('-ch', '--checkpoint', metavar='DIR', help='path to checkpoint (default: ./checkpoint)', default='./checkpoint')
parser.add_argument('-lg', '--log', metavar='DIR', help='path to log (default: ./log)', default='./log')
parser.add_argument('-ds', '--dataset', metavar='FILE', help='dataset to use [cifar100, nuswide, coco, cocosent] (default: cifar10)', default='cifar100')
parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-st', '--step_size', default=10, type=int, metavar='N', help='step size to decay the learning rate (default: 10)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N', help='number of workers for data processing (default: 4)')
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('-sz', '--image_size', default=32, type=int, metavar='N', help='Size of input to use (default: 32)')
parser.add_argument('-c', '--channels', default=3, type=int, metavar='N', help='Number of channels of the input, which could be different for sentences (default: 3)')
parser.add_argument('-nb', '--num_class', default=100, type=int, metavar='N', help='Number of binary bits to train (default: 8)')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

args = parser.parse_args()
pprint(args)

# check and create directories
if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)

if not os.path.exists(args.log):
    os.makedirs(args.log)    


arch = 'ConvNet_'
filename = arch + args.dataset+'_'+str(args.num_class)
checkpoint_filename = os.path.join(args.checkpoint, filename+'.pt')


model = CNN(num_class=args.num_class)
criterion = torch.nn.CrossEntropyLoss(size_average=True)
weight_criterion = CE(aggregate='sum')

use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.cuda()
    criterion = criterion.cuda()
    weight_criterion.cuda()
    torch.cuda.manual_seed(args.seed)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Adjust learning rate and betas for Adam Optimizer
n_epoch=200
epoch_decay_start = 80
learning_rate = 1e-3
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * n_epoch
beta1_plan = [mom1] * n_epoch
for i in range(epoch_decay_start, n_epoch):
    alpha_plan[i] = float(n_epoch - i) / (n_epoch - epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1



def save_checkpoint(state, is_best, prefix='', filename='./checkpoint/checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        print("====> saving the new best model")
        path = "/".join(filename.split('/')[:-1])
        best_filename = os.path.join(path, prefix+'model_best'+'.pt')
        shutil.copyfile(filename, best_filename)


def rampup(global_step, rampup_length=200):
    if global_step <rampup_length:
        global_step = np.float(global_step)
        rampup_length = np.float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return np.exp(-5.0 * phase * phase)



def generate_weight(trainLabels, predictLabels, epoch, b_epoch, s0=2, gamma_d=30, total_epochs=200):
    pred_weights = np.squeeze(np.zeros((trainLabels.shape[0],1)))

    u_labels = np.squeeze(np.unique(trainLabels))

    
    for iter in range(u_labels.shape[0]):
        index = np.squeeze(np.where(trainLabels==u_labels[iter]))
        tep_weight = np.squeeze(predictLabels[index, u_labels[iter]])
        idx = np.argsort(tep_weight)

        s_idx = np.squeeze(np.where(idx[tep_weight>=0.5]))


        if epoch<=b_epoch:
            if s_idx.size < np.int(np.int(0.1*index.shape[0])):
                start_idx = index.shape[0]- np.int(np.int(0.1*index.shape[0]))
                s0 = np.int(np.floor(10*s_idx.size/index.shape[0])+1)

                st = (total_epochs-30-b_epoch)//max(10-s0,1)
                gamma = (10-s0)*gamma_d
            else:
                start_idx = index.shape[0]- np.int(np.floor(10*s_idx.size/index.shape[0])+1)*np.int(np.int(0.1*index.shape[0]))
                s0 = np.int(np.floor(10*s_idx.size/index.shape[0])+1)
                if s0>5:
                    s0=5
                st = (total_epochs-30-b_epoch)//max(10-s0,1)
                gamma = (10-s0)*gamma_d
        else:
            st = (total_epochs-30-b_epoch)//max(10-min(s0,5),1)
            gamma = (10-min(s0,5))*gamma_d
            start_idx = index.shape[0]- (min(s0,5)*np.int(np.int(0.1*index.shape[0]))+ min(max(epoch-b_epoch, 0)//st, 10-min(s0,5))*np.int(0.1*index.shape[0]))
            if start_idx<0:
                start_idx = 0


        pred_weights[index[idx[start_idx:]]] = 1.0
    
    if epoch<b_epoch:
        return pred_weights, s0, gamma
    else:
        return pred_weights

def generate_data_weight(model, dataloader_test, epoch, b_epoch, s0=2, gamma_d=30, use_gpu=True):
    trainLabels = []
    predictLabels = []
    model.eval()

    for iteration, data in enumerate(dataloader_test, 0):
        inputs, labels, index = data['image'], data['labels'], data['index']
        
        if inputs.size(3)==3:
            inputs = inputs.permute(0,3,1,2)
        
        inputs = inputs.type(torch.FloatTensor)

        trainLabels.extend(labels.numpy())

        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
    

        y, _= model(inputs)

        pred_y = F.softmax(y, dim=1)
        predictLabels.extend(pred_y.data.cpu().numpy())

    trainLabels = np.squeeze(np.array(trainLabels))
    predictLabels = np.array(predictLabels)

    if epoch<b_epoch:
        train_weights, s0, gamma= generate_weight(trainLabels, predictLabels, epoch, b_epoch, s0=s0, gamma_d=gamma_d)
        
        return torch.from_numpy(train_weights).float(), s0, gamma
    else:
        train_weights = generate_weight(trainLabels, predictLabels, epoch, b_epoch, s0=s0, gamma_d=gamma_d)    
        
        return torch.from_numpy(train_weights).float()

def masked_crossentropy(out, labels, mask_flag):
    cond = (mask_flag> 0)
    nnz = torch.nonzero(cond)
    nbsup = len(nnz)
    # check if labeled samples in batch, return 0 if none
    if nbsup > 0:
        masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
        masked_labels = labels[cond]
        loss = F.cross_entropy(masked_outputs, masked_labels)
        return loss
    return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False)


def train(dataloader, test_loader, dataloader_test, true_label, noisy_rate):
    start_epoch=0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    best_prec = -99999


    epoch_test_accuracy = np.squeeze(np.zeros((args.epochs,1)))

    n_samples=50000
    
    s0=2
    st=10
    gamma_d = 50
    gamma =50 
    b_epoch=40      # It is T1 in the paper
    
    pred_weights = torch.squeeze(torch.zeros(n_samples, 1).float())   # It determines whether to add the sample into curriculum
    p_labels =  torch.squeeze(torch.zeros(n_samples,args.num_class).float()) # It is p^{t-1}

    s_a = np.squeeze(np.ones((b_epoch,1)))

    for epoch in range(start_epoch, args.epochs):      
        model.train()

        running_loss = 0.0

        if epoch<b_epoch:
            u_w = 0.0    
        else:
            u_w = rampup(epoch-b_epoch,rampup_length=args.epochs-b_epoch)

        u_w_m = gamma*u_w

        u_w_m = torch.autograd.Variable(torch.FloatTensor([u_w_m]).cuda(), requires_grad=False)

        for iteration, data in enumerate(dataloader, 0):

            data_time.update(time.time() - end)
            inputs, labels, index = data['image'], data['labels'], data['index']

            r_targets = torch.squeeze(torch.arange(0, args.num_class))
            r_targets = r_targets.repeat(index.size(0),1).view(args.num_class*index.size(0),-1)
            
            r_targets = r_targets.long()

            if inputs.size(3)==3:
                inputs = inputs.permute(0,3,1,2)
            
            inputs = inputs.type(torch.FloatTensor)

            targets = labels.type(torch.LongTensor)


            z_comp = pred_weights[index]
            b_comp = p_labels[index,:]

            if use_gpu:
                inputs = Variable(inputs.cuda())
                targets = Variable(targets.cuda())
                z_comp = Variable(z_comp.cuda(), requires_grad=False)
                b_comp = Variable(b_comp.cuda(), requires_grad=False)
                r_targets = Variable(r_targets.cuda(), requires_grad = False)

            else:
                inputs = Variable(inputs)
                targets = Variable(targets)
                z_comp = Variable(z_comp, requires_grad=False)
                b_comp = Variable(b_comp, requires_grad=False)
                r_targets = Variable(r_targets, requires_grad = False)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            

            y, _= model(inputs)


            if epoch>b_epoch:
                beta = torch.squeeze(b_comp.view(b_comp.size(0)*b_comp.size(1), -1))
                r_Y = y.repeat(1,args.num_class).view(y.size(0)*args.num_class,y.size(1))

                loss = masked_crossentropy(y, targets, z_comp) + u_w_m*weight_criterion(r_Y, r_targets, weights=beta)/beta.sum()
            else:
                loss = F.cross_entropy(y, targets, size_average=True)
        
            p_labels[index,:] = F.softmax(y.data.clone().cpu(), dim=1)

            # backward+optimize
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            
            end = time.time()
        

        print("Epoch[{}]({}/{}): Time:(data {:.3f}/ batch {:.3f}) Loss_H: {:.4f}".format(epoch, iteration, len(dataloader), 
                    data_time.val, batch_time.val, loss.item()))
        adjust_learning_rate(optimizer, epoch)

        if epoch<b_epoch:
            pred_weights, s0, gamma = generate_data_weight(model, dataloader_test, epoch, b_epoch, s0=s0, gamma_d=gamma_d, use_gpu=True)
            s_a[epoch]=s0
        else:
            if epoch==b_epoch:
                s0 = np.int(np.amax(s_a))
                gamma = (10-min(s0,5))*gamma_d 
            if epoch % 20==0:
                print('s0 is:{}'.format(s0))
                print('gamma is:{}'.format(gamma))
            pred_weights = generate_data_weight(model, dataloader_test, epoch, b_epoch, s0=s0, gamma_d=gamma_d, use_gpu=True)

        
        Accuracy = du.predict(model, test_loader)
        print("Test image accuracy is:{}".format(Accuracy))
        
        epoch_test_accuracy[epoch] = Accuracy

        is_best = Accuracy > best_prec
        best_prec = max(best_prec, Accuracy)
        save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec' : best_prec,
                'optimizer' : optimizer.state_dict(),
                }, is_best, prefix=arch, filename=checkpoint_filename)

    
    io.savemat('cifar100_pair_Conv_epoch_test_accuracy'+'_'+ str(noisy_rate)+'.mat', mdict={'test_accuracy': epoch_test_accuracy})




def main():
    print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])

    transforms_test = transforms.Compose([
        transforms.ToTensor()
        ])
    

    mode = {'train': True, 'test': True}


    rate = np.squeeze([0.45,0.4])

    for iter in range(1):
        
        image_datasets ={'train': Cifar100(train=True, transform = None),
                      'test': Cifar100(train=False, transform = None)}


        trainData = image_datasets['train'].x
        trainLabel = image_datasets['train'].y


        testData = image_datasets['test'].x
        testLabel = image_datasets['test'].y
        
        true_label = np.squeeze(trainLabel).copy()


        trainLabel, actual_noise_rate = GN.noisify(nb_classes=args.num_class, train_labels=np.squeeze(trainLabel), noise_type='pairflip', noise_rate=rate[iter])
        

        trainData = np.array(trainData)
        trainLabel = np.squeeze(trainLabel)
        testData = np.array(testData)
        testLabel = np.squeeze(testLabel)

        
        train_data = DT(trainData= trainData, trainLabel = trainLabel, transform=transforms_train)
        train_data_test = DT(trainData= trainData, trainLabel = trainLabel, transform=transforms_test)
        test_data = DT(trainData= testData, trainLabel = testLabel, transform=transforms_test)

        train_loader =  torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle=True, num_workers=args.workers)
        train_loader_test =  torch.utils.data.DataLoader(train_data_test, batch_size = args.batch_size, shuffle=False, num_workers=args.workers)
        
        test_loader =  torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, shuffle=False, num_workers=args.workers)




        train(train_loader, test_loader, train_loader_test, true_label, rate[iter])
        
    

if __name__ == "__main__":
    sys.exit(main())