from TDNN_gpu import *
import numpy as np
import time
import os
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data.dataset import Dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



mfcc_train="/media/dream/新加卷/ashell1Data/dataMfcc/train"
mfcc_enroll="/media/dream/新加卷/ashell1Data/dataMfcc/enroll"
mfcc_test="/media/dream/新加卷/ashell1Data/dataMfcc/test"


speakers=os.listdir(mfcc_train)
train_mfcc=[]
train_label=[]
class MyCustomDataset(Dataset):
    _lab = []
    def __init__(self):
        with open('dataMfccTxt/trainLabel.txt', 'r') as f:
            line = f.readline()
            while (line):
                line = line.strip().split(' ')
                c = [item for item in line]
                self._lab.append(c)
                line = f.readline()

    def __getitem__(self, item):
        temp = np.load(mfcc_train + "/" + self._lab[item][1])
        temp = temp[10:180, :]
        return [temp,int(self._lab[item][2])]

    def __len__(self):
        return len(self._lab)

train_dataset = MyCustomDataset()
train_dataset_loader=data.DataLoader(train_dataset,shuffle=False, batch_size=1)

def test_right():
    net = torch.load('total_model_new25')
    net.eval()
    i=0
    numCorrect=0
    for s, data in enumerate(train_dataset_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.type(torch.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda()
        output = net(inputs).cuda()
        z=torch.sum(torch.max(output, 1)[1] - labels == 0).float()/1
        print(z)
        numCorrect=numCorrect+z
        i=i+1
    print(numCorrect/i)
    print()


if __name__ == '__main__':
    test_right()