# -*- coding: utf-8 -*-
"""Nov12_2020_AE_umap.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
# %reset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import randn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import time
import pickle
from datetime import datetime
!pip install hickle
import hickle as hkl
from torch.autograd import Variable
import gzip
import sys
import os 
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD
torch.set_default_tensor_type(torch.DoubleTensor)

#https://github.com/deeptools/pyBigWig
!pip install pyBigWig
import pyBigWig

"""## Load data"""

class JointDataset():
    def __init__(self, data, mode='train'):
        self.mode = mode
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img = self.data[index]
        return img

# ['Dnase','H3k27ac','H3k4me3','H3k27me3','Ctcf','Smc3']

# chromosome 1 to 10
n_samples_chr = 2000
x_list, y_list = [], []
# for chrom in ["chr"+str(i) for i in range(1,6)]:
for chrom in range(1,6):
  print(chrom,datetime.now())
  with open("/training_testing/Nov4_GM12878_"+str(chrom)+"_3000samples_vertical.txt","rb") as fp:
    sublist = pickle.load(fp)
  x_list += sublist[0][:n_samples_chr]
  y_list += sublist[1][:n_samples_chr]

all_train = JointDataset([[i,j] for i,j in zip(x_list,y_list)])
del x_list, y_list

class HiAE(nn.Module):
  def __init__(self, input_channel = 6, inter_channel = 6, output_channel = 6, channel_ntimes = 8):
    super(HiAE, self).__init__()
    self.input_channel = input_channel
    self.output_channel = output_channel
    self.channel_ntimes = channel_ntimes
    self.inter_channel = inter_channel
    self.encoder = nn.Sequential(
        nn.Conv1d(in_channels=self.input_channel, out_channels=self.input_channel*self.channel_ntimes, kernel_size=100, stride=5,padding=5),
        nn.BatchNorm1d(self.input_channel*self.channel_ntimes),
        nn.ReLU(),
        nn.Conv1d(in_channels=self.input_channel*self.channel_ntimes, out_channels=self.input_channel*self.channel_ntimes, kernel_size=50, stride=5,padding=5),
        nn.BatchNorm1d(self.input_channel*self.channel_ntimes),
        nn.ReLU(),
        nn.Conv1d(in_channels=self.input_channel*self.channel_ntimes, out_channels=self.inter_channel, kernel_size=10, stride=4,padding=10)
    )
    self.decoder = nn.Sequential(
        nn.ConvTranspose1d(in_channels=self.inter_channel, out_channels=self.inter_channel*self.channel_ntimes, kernel_size=10, stride=4,padding=8),
        nn.BatchNorm1d(self.input_channel*self.channel_ntimes),
        nn.ReLU(),
        nn.ConvTranspose1d(in_channels=self.inter_channel*self.channel_ntimes, out_channels=self.inter_channel*self.channel_ntimes, kernel_size=50, stride=5,padding=6),
        nn.BatchNorm1d(self.inter_channel*self.channel_ntimes),
        nn.ReLU(),
        nn.ConvTranspose1d(in_channels=self.inter_channel*self.channel_ntimes, out_channels=self.output_channel, kernel_size=100, stride=5,padding=5)
    )

  def forward(self,x):
    x = torch.log2(x+1)
    x_enc = self.encoder(x)
    x_out = self.decoder(x_enc)
    return x_out

x,y = all_train[0]
model = HiAE()
pth = "/models/Nov9_Gm12878_verticalHiC_MSE_zscore_log2_chr1to5_6chip_HiCAE_6enc.pt"
model.load_state_dict(torch.load(pth))
print(model(x.transpose(1,0)).shape)

k = np.random.randint(0,len(all_train),1)[0]
x = all_train[k][0]
fig,axarr = plt.subplots(1,6,constrained_layout=True,figsize=(30,5))
res = model(x.transpose(1,0))
elements = ['Dnase','H3k27ac','H3k4me3','H3k27me3','Ctcf','Smc3']
for i in range(x.shape[0]):
  axarr[i].scatter(np.log2(x[i,0,:].detach().numpy()+1),res[0,i,:].detach().numpy())
  axarr[i].set_title(elements[i],fontsize=20)
  axarr[i].set_xlabel("Original signal",fontsize=20)
  axarr[i].set_ylabel("Reconstructed signal",fontsize=20)
fig.suptitle("Simple auto-encoder reconstruction of ChIP-seq signals",fontsize=25)
plt.show()