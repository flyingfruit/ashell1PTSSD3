from TDNN_gpu import *
import numpy as np
import time
import os
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
import model.discriminator
from ModelDANN.DANN import DANN
from ModelDANN.DANN2 import DANN2

mfcc_train="/media/dream/新加卷/ashell1Data/dataMfcc/train"
mfcc_enroll="/media/dream/新加卷1/subCN-Celeb/enrollMfccnpy"
mfcc_test="/media/dream/新加卷1/subCN-Celeb/testMfccnpy"
dataLabel='../dataMFCCtxt/trainLabel.txt'
dataLabel2='../dataMFCCtxt/TestSubLonglabel.txt'
frameLength=300
frameLength2=300

class MyCustomDataset(Dataset):
    _lab = []

    def __init__(self):
        with open(dataLabel, 'r') as f:
            line = f.readline()
            while (line):
                line = line.strip().split(' ')
                c = [item for item in line]
                # temp = np.load(mfcc_train + "/" + c[1])
                # print(temp.shape)
                line = f.readline()
                self._lab.append(c)

    def __getitem__(self, item):
        temp = np.load(mfcc_train + "/" + self._lab[item][1])
        temp = temp[10:frameLength, :]
        return [temp, int(self._lab[item][2])]

    def __len__(self):
        return len(self._lab)


# class MyCustomDataset2(Dataset):
#     _lab = []
#
#     def __init__(self):
#         with open(dataLabel2, 'r') as f:
#             line = f.readline()
#             while (line):
#                 line = line.strip().split(' ')
#                 c = [item for item in line]
#
#                 self._lab.append(c)
#                 line = f.readline()
#
#     def __getitem__(self, item):
#         temp = np.load(mfcc_train + "/" + self._lab[item][1])
#         temp = temp[10:frameLength, :]
#         print(temp.shape)
#         return [temp,int(self._lab[item][2])]
#
#     def __len__(self):
#         return len(self._lab)

class MyCustomDataset2(Dataset):
    _lab = []

    def __init__(self):
        with open(dataLabel2, 'r') as f:
            line = f.readline()
            while (line):
                line = line.strip().split(' ')
                c = [item for item in line]
                line = f.readline()
                # if(not os.path.exists(mfcc_enroll + "/" + c[1])):
                #     continue
                temp = np.load(mfcc_enroll + "/" + c[1])
                print(temp.shape[0],":",c[1])
                if (temp.shape[0]-10 < frameLength2):
                    print("not")
                    continue
                self._lab.append(c)

    def __getitem__(self, item):
        temp = np.load(mfcc_enroll + "/" + self._lab[item][1])
        temp = temp[10:frameLength2, :]
        return [temp,int(self._lab[item][2])]

    def __len__(self):
        return len(self._lab)


train_mfccs={}
speakers=os.listdir(mfcc_train)

source_dataset = MyCustomDataset()
source_dataset_loader=data.DataLoader(source_dataset,shuffle=True, batch_size=100)
target_dataset= MyCustomDataset2()
target_dataset_loader=data.DataLoader(target_dataset,shuffle=True, batch_size=100)


print(target_dataset.__len__())
print(source_dataset.__len__())

def train(num):
    domain_model=DANN2()
    domain_model.cuda()
    domain_loss=nn.CrossEntropyLoss()
    task_loss=nn.CrossEntropyLoss()
    lr=0.001
    optimizer=torch.optim.Adam(domain_model.parameters(),lr=lr)
    epochs=200
    for epoch in range(50,epochs):
        for i,data in enumerate(target_dataset_loader,0):
            for j,data2 in enumerate(source_dataset_loader,0):
                print("start:",epoch)
                inputsT,labelsT=data
                inputsS,labelsS=data2
                inputsS=inputsS.type(torch.FloatTensor)
                inputsT=inputsT.type(torch.FloatTensor)
                inputsS,labelsS=inputsS.cuda(),labelsS.cuda()
                p=epoch/epochs
                alpha=torch.tensor(2./(1.+np.exp(-10*p))-1)
                src_pre,src_domain,_=domain_model(inputsS,alpha)
                inputsT, labelsT =inputsT.cuda(),labelsT.cuda(),
                _,dst_domain,_=domain_model(inputsT,alpha)
                # print(feature[1])
                src_labelloss=task_loss(src_pre,labelsS)
                src_domain_loss = domain_loss(src_domain, torch.ones(len(src_domain)).long().cuda())
                dst_domain_loss = domain_loss(dst_domain, torch.zeros(len(dst_domain)).long().cuda())
                losses=src_labelloss+src_domain_loss+dst_domain_loss
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                print("end:",epoch)
                # print(domain_model)
                break
        torch.save(domain_model.state_dict(), '../model/domainmodel{}'.format(epoch))



def test():

    # domain_model = DANN2()
    # domain_model.load_state_dict(torch.load('../model/domainmodel60'))
    domain_model=torch.load("../total_model_new311")[:7]
    print(domain_model)
    # exit()
    domain_model.eval()

    # domain_model2 = DANN2()
    # domain_model2.load_state_dict(torch.load('../model/domainmodel60'))
    # domain_model2.eval()
    with open("../dataMFCCtxt/TestSubLonglabel2.txt", 'r') as f:
        line = f.readline()
        while (line):
            line = line.strip().split(' ')
            c = [item for item in line]
            line = f.readline()
            temp = np.load(mfcc_test+ "/" + c[1])
            # temp2=temp
            if(2000>temp.shape[0]):
                mid=temp.shape[0]
            else:
                mid=2000
            temp=temp[:mid]
            # print(temp.shape)temp=temp.cuda()
            temp=torch.from_numpy(temp)
            temp=temp.reshape(1,temp.shape[0],temp.shape[1])
            temp=temp.type(torch.FloatTensor)
            # print(temp.shape)
            temp=temp.cuda()
            # temp2=torch.from_numpy(temp2)
            # temp2 = temp2.reshape(1, temp2.shape[0], temp2.shape[1])
            # temp2 = temp2.type(torch.FloatTensor)
            # temp2 = temp2.cuda()
            feature = domain_model(temp)
            # _, dst_domain, feature2 = domain_model2(temp2, 1)
            # print(feature)
            # print(feature2)
            # exit()
            print(c[1])
            # _, dst_domain, feature = domain_model(temp, 1)
            # print(feature)
            feature=feature.cpu().float().detach().numpy()
            # feature2 = feature2.cpu().float().detach().numpy()
            # np.save("/media/dream/新加卷/ashell1Data/DANNXvectorC62/enroll/{}".format(c[1]),feature2)
            np.save("/media/dream/新加卷/ashell1Data/DANNXvectorC62/test/{}".format(c[1]),feature)
    # for i, data in enumerate(target_dataset_loader, 0):
    #     inputsT, labelsT = data
    #     print(inputsT.shape)
    #     inputsT = inputsT.type(torch.FloatTensor)
    #     inputsT, labelsT = inputsT.cuda(), labelsT.cuda()
    #     _, dst_domain,feature = domain_model(inputsT, 1)
    #     print(feature[0])

if __name__ == '__main__':
    test()
    # train(1)
    # adapter2()