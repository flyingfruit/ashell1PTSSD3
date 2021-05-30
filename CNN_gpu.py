import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import math

class StatsPooling(nn.Module):
    def __init__(self):
        super(StatsPooling,self).__init__()

    def forward(self,varient_length_tensor):
        mean = varient_length_tensor.mean(dim=1)
        std = varient_length_tensor.std(dim=1)
        return torch.cat((mean,std),dim=1)

class TCNN(nn.Module):
    def __init__(self, input_dim, output_dim, keral,batchnorm, device='cpu'):
        super(TCNN,self).__init__()
        self.batchnorm=batchnorm
        self.device=device
        cnn1=nn.Conv1d(input_dim, output_dim, keral)
    def forward(self, x):
        x=self.cnn1(x)
        if self.batchnorm:
            x=nn.BatchNorm1d