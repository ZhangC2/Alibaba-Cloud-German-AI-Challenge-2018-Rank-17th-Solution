import os
import pdb
import h5py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transform
from os.path import join


class AverageMeter(object):
    def __init__(self, channels = 1):
        self.channels = channels
        self.reset()
        
    def reset(self):
        self.val = np.zeros((self.channels,), dtype=np.float32)
        self.avg = np.zeros((self.channels,), dtype=np.float32)
        self.sum = np.zeros((self.channels,), dtype=np.float32)
        self.count = np.zeros((self.channels,), dtype=np.float32)

    def update(self, val):
        self.val = val
        self.sum += val 
        self.count += 1
        self.avg = self.sum / self.count


img_dir = '../datafloder/train-img/'
images = os.listdir(img_dir)
images.sort()
print(len(images))
mean = AverageMeter(channels=3)
sum1 = np.zeros((3,), dtype=np.float32)
mean_=[0.4922792,0.4634237,0.3974504]
ToTensor = transform.ToTensor()

for i in range(len(images)):    
    img = Image.open(join(img_dir, images[i]))
    img = ToTensor(img)
    print(i)
    for ii in range(3):        
        sum1[ii] = (torch.mean(img[ii,:,:])-mean_[ii])**2    
    mean.update(sum1)
#meanavg = np.sqrt(mean.avg)   
print (mean.avg)