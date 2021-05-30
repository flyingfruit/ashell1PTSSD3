import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# n=np.array([[[1,-1,0],[1,-1,0],[1,-1,0],[1,-1,0],[1,-1,0],[1,-1,0]],[[5,1,3],[5,1,3],[5,1,3],[5,1,3],[5,1,3],[5,1,3]]],dtype='float')
n=np.array([[[1,0,0,0],[2,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]],dtype='float')
input=torch.from_numpy(n).type(torch.FloatTensor)
input=input
m=nn.BatchNorm1d(6)
print(input)
output=m(input)
print(output.shape)
print(output)