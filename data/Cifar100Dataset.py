from __future__ import print_function, division
import os
import sys
import torch
import _pickle as cPickle
import tarfile
import numpy as np
from pprint import pprint
from six.moves import urllib
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

def maybe_download_and_extract(dest_directory = './data/cifar100'):
    """Download and extract the cifar100 dataset."""

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) /
                              float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    
    return dest_directory

class Cifar100Dataset(Dataset):
    """Cifar100 Landmarks dataset."""

    def __init__(self, train=True, transform=None):
        """
       
        """
        path = maybe_download_and_extract()

        self.root_dir = path

        if train:
            fpath = os.path.join(path, 'cifar-100-python', 'train')
            x, y = self.__load_batch(fpath)
        else:
            fpath = os.path.join(path, 'cifar-100-python', 'test')
            x, y = self.__load_batch(fpath)

        # convert to NXCxHXW
        self.x = x.transpose(0, 2, 3, 1)
        self.y = y[:,1]
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        img, target = self.x[index], self.y[index][1]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        sample = {'image': img, 'labels': target, 'index': index}

        return sample

    def __load_batch(self, fpath):
        """Internal utility for parsing CIFAR data.
        # Arguments
            fpath: path the file to parse.
        # Returns
            A tuple `(data, labels(coarse_labels, fine_labels))`.
        """
        f = open(fpath, 'rb')
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
        f.close()
        data = d['data']
        fine_labels = d["fine_labels"]
        coarse_labels = d["coarse_labels"]
        labels = np.column_stack((coarse_labels, fine_labels))

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels
    
    
    