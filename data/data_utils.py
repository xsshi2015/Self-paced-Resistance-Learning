from __future__ import print_function, division
import numpy as np
import torch
from torch.autograd import Variable
import argparse
import os
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch Deep Noisy Label')
parser.add_argument('-ch', '--checkpoint', metavar='DIR', help='path to checkpoint (default: ./checkpoint)', default='./checkpoint')
parser.add_argument('-w', '--workers', default=0, type=int, metavar='N', help='number of workers for data processing (default: 4)')
parser.add_argument('-nb', '--num_bits', default=10, type=int, metavar='N', help='Number of binary bits to train (default: 8)')


def produce_noisydata(Data, Label, rate=0.2):
    
    u_label = np.squeeze(np.unique(Label))

    for iter in range(u_label.size):
        idx = np.squeeze(np.where(Label==u_label[iter]))
        sn = int(rate*idx.size)
        s_idx = np.squeeze(np.random.choice(idx, sn, replace=False))
        r_label = np.setdiff1d(u_label, u_label[iter])
        Label[s_idx] = np.squeeze(np.random.choice(r_label, sn, replace=True))
    
    return Data, Label

def split_idx(train_labels, select_num=1000):
    trainLabel = np.squeeze(train_labels)

    u_label = np.squeeze(np.unique(trainLabel))
    train_idx = []
    dict_idx = []

    for iter in range(u_label.size):
        idx = np.squeeze(np.where(trainLabel==u_label[iter]))
        sn = int(select_num/u_label.size)
        s_idx = np.squeeze(np.random.choice(idx, sn, replace=False))
        dict_idx.extend(s_idx)

        r_idx = np.squeeze(np.setdiff1d(idx, s_idx))
        train_idx.extend(r_idx)

    train_idx = np.squeeze(train_idx)
    dict_idx = np.squeeze(dict_idx)
    # import pdb; pdb.set_trace()

    return train_idx, dict_idx

def predict(model, dataloader, batch_size=50, use_gpu=True):
    
    num_train_batch = int(10000/batch_size)

        
    total = 0
    correct = 0 

    model.eval()
    # Some stats to monitor the loss
    for iteration, data in enumerate(dataloader, 0):
        if iteration == num_train_batch:
            break

        #data = next(iter(dataloaders['val']))
        inputs, labels, _ = data['image'], data['labels'], data['index']
        # import pdb; pdb.set_trace()
        if inputs.size(3)==3:
            inputs = inputs.permute(0,3,1,2)
            inputs = inputs.type(torch.FloatTensor)

        labels = labels.type(torch.LongTensor)
        # print("Collecting values for train data[{}/{}]".format(iteration, num_train_batch-1))

        if use_gpu:
            labels = Variable(labels.cuda())
        else:
            labels = Variable(labels)

        with torch.no_grad():
            inputs = Variable(inputs.cuda())
            y, _= model(inputs)

        _, predicted = torch.max(y.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
    
    Accuracy = 100.*np.float(correct)/total

    return Accuracy



