from TDNN_gpu import *
import random
import numpy as np
import time
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import copy
import torch.utils.data as data


mfcc_train="/media/dream/新加卷/ashell1Data/dataMfcc/train"
mfcc_test="/media/dream/新加卷/ashell1TF/dataMfcc/test"
enrollDir='/media/dream/新加卷/ashell1Data/dataMfcc/enroll'
dataLabel='./dataMfccTxt/trainSubLongLabel.txt'
enrollLabel='dataMfccTxt/enrollSubLongLabel.txt'
VoxenrollDir='/media/dream/新加卷/ashell1Data/dataMfcc/voxenroll'
VoxenrollLabel="dataMfccTxt/VoxenrollLabel.txt"
frameLength=300


class MyCustomDataset():
    _lab = []
    _l={}#说话人地址信息
    _len=-1#多少说话人
    _llen={}#每个说话人音频数
    def __init__(self):
        # with open(dataLabel, 'r') as f:
        with open(VoxenrollLabel, 'r') as f:  # change enroll
        # with open(enrollLabel, 'r') as f:# change enroll
            line = f.readline()
            firstLab={}
            # temp=-1
            while (line):
                line = line.strip().split(' ')
                c = [item for item in line]
                self._lab.append(c)
                if int(c[2])>self._len:
                    self._len=int(c[2])
                line = f.readline()
            for i in range(self._len+1):
                self._l.update({i:[]})
                self._llen.update({i:0})
            for _,it,label in self._lab:
                self._l[int(label)].append(it)
                self._llen[int(label)]+=1

    def getitem(self,p,k):
        spes = random.sample(range(0, self._len), p)
        mfccs={}
        for speaker in spes:
            mfccs.update({speaker:[]})
            l = random.sample(range(0, self._llen[speaker]), k)
            for i in l:
                # mfccs[speaker].append(np.load(mfcc_train + "/" + self._l[speaker][i]))
                mfccs[speaker].append(np.load(VoxenrollDir + "/" + self._l[speaker][i]))  # change enroll
                # mfccs[speaker].append(np.load(enrollDir + "/" + self._l[speaker][i]))# change enroll
        return mfccs,spes


def get_minP_maxN(y,person,ker):
    clases_dict = {}
    result=cosine_similarity(y,y)
    for i in range(person*ker):
        xu=i//ker
        positive=result[i,xu*ker:xu*ker+ker]
        begin=result[i,:xu*ker]
        end=result[i,xu*ker+ker:]
        negitave=np.hstack((begin,end))
        minP=np.argmin(positive)+ker*xu
        maxN=np.argmax(negitave)
        if maxN>=xu*ker:
            maxN=maxN+ker
        clases_dict[i]=[minP,maxN]
    return clases_dict

def tripleLoss_in_l2():
    net = torch.load('total_model_new311')[:-3]
    dataLoader = MyCustomDataset()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(100):
        mfcc, label = dataLoader.getitem(6, 4)
        n1=np.random.rand(1,512)
        inputs=[]
        for speakersMfcc in mfcc.values():
            for speakerMfcc in speakersMfcc:
                inputs.append(torch.from_numpy(speakerMfcc[np.newaxis,:]).cuda().float())
                output=net(torch.from_numpy(speakerMfcc[np.newaxis,:]).cuda().float())
                temp= output.cpu().detach().numpy()
                n1=np.append(n1,temp,axis=0)
        n1=n1[1:,:]
        clases_dict=get_minP_maxN(n1,6,4)
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        for r,b in clases_dict.items():
            temp = inputs[r]
            anchor=net(temp)
            temp = inputs[b[0]]
            positive=net(temp)
            temp = inputs[b[1]]
            negative=net(temp)
            optimizer.zero_grad()
            out=triplet_loss(anchor,positive,negative)
            print(out)
            out.backward()
            optimizer.step()
        print("epoch:{}".format(epoch))
        torch.save(net, 'total_model_trilp{}'.format(epoch))
    # print(net_new)
    pass

if __name__ == '__main__':
    # test()
    tripleLoss_in_l2()
    # net = torch.load('total_model')[:-1]
    # print(net)
    pass