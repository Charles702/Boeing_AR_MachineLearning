from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import PIL.Image as Image
import cv2
import pyredner
import torch
import redner
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh.transformations as transformations
import os
import h5py
import math
from matplotlib.pyplot import figure
from PIL import Image
from torchsummary import summary
import pandas as pd
import numpy as np


# generate samples in [lb,ub] interval according to guassian distribution 
def make_normal_bounded(lb, ub, nsamples, mu=None, sigma=None):
    """Draw samples from a normal distribution with enforced lower and upper bound."""
    lu_range = ub - lb
    def in_range(x):
        return x >= lb and x <= ub

    if mu is None:
        mu = lb + 0.5 * lu_range
    if sigma is None:
        sigma = lu_range * .5
    samples = []
    # s = np.random.normal(mu, sigma, nsamples)
    # s = list(filter(in_range, s))
    # samples.extend(s)
    while len(samples) < nsamples:
        s = np.random.normal(mu, sigma, nsamples)
        s = list(filter(in_range, s))
        n_add = max(nsamples - len(samples), 0)
        samples.extend(s[:n_add])

    return samples


class AnchorDataset(Dataset):
  def __init__(self, dir_h5, dstfile_h5, transform=None, applySeg=True, greyScale=False):
    # training_s_dir: the directory which stored the sourece images
    # training_t_dir: the directory which stored the target images
    self.transform = transform
    self.img_paths = []
    self.pose_labels = []
  
    #read data from h5 file
    file_p = os.path.join(dir_h5, dstfile_h5)
    self.file_dataset = h5py.File(file_p, 'r')
          
  def __len__(self):
      return self.file_dataset['Train_im_t'].shape[0]

  def __getitem__(self, index):
    #get image path
    # convert target img name
    print(index)

    img_t = self.file_dataset['Train_im_t'][index]
    img_s = self.file_dataset['Train_im_s'][index]
    pose_t = self.file_dataset['Train_p_t'][index]
    pose_s = self.file_dataset['Train_p_s'][index]  

    if self.transform is not None:
      img_t_input = self.transform(Image.fromarray(img_t.astype('uint8')))
      img_s_input = self.transform(Image.fromarray(img_s.astype('uint8')))
    
    return img_t_input, img_s_input, pose_t, pose_s

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import PIL.Image as Image
import cv2

# shrink size of h5 dataset by removing repeated target images
class AnchorDataset_shrink(Dataset):
  def __init__(self, dir_h5, dstfile_h5, s_size, transform=None, applySeg=True, greyScale=False):
    # training_s_dir: the directory which stored the sourece images
    # training_t_dir: the directory which stored the target images
    # s_size: number of source image generated from a target image

    self.transform = transform
    self.img_paths = []
    self.pose_labels = []
  
    #read data from h5 file
    file_p = os.path.join(dir_h5, dstfile_h5)
    self.file_dataset = h5py.File(file_p, 'r')
          
  def __len__(self):
      return self.file_dataset['Train_im_s'].shape[0]

  def __getitem__(self, index):
    #get image path
    # egt index of target images
    tg_index = index//s_size
    img_t = self.file_dataset['Train_im_t'][tg_index]

    img_s = self.file_dataset['Train_im_s'][index]
    pose_t = self.file_dataset['Train_p_t'][index]
    pose_s = self.file_dataset['Train_p_s'][index]  

    if self.transform is not None:
      img_t_input = self.transform(Image.fromarray(img_t.astype('uint8')))
      img_s_input = self.transform(Image.fromarray(img_s.astype('uint8')))
    
    return img_t_input, img_s_input, pose_t, pose_s

       