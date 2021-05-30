from modelC.CModel import CYModel
from modelC.BModel import BYModel
from model.discriminator import Discriminator
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.dataset import Dataset

def trainGener():
    pass


frameLength=300
frameLength2=300

dataLabel="../dataMFCCtxt/trainSubLabelforCycleGanF.txt"
dataLabel2="../dataMFCCtxt/TestSubLonglabelF.txt"
mfcc_train="/media/dream/新加卷/ashell1Data/dataMfcc/train"
mfcc_enroll="/media/dream/新加卷1/subCN-Celeb/enrollMfccnpy"
mfcc_test="/media/dream/新加卷1/subCN-Celeb/testMfccnpy"


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
                self._lab.append(c)

    def __getitem__(self, item):
        temp = np.load(mfcc_enroll + "/" + self._lab[item][1])
        temp = temp[10:frameLength2, :]
        return [temp,int(self._lab[item][2])]

    def __len__(self):
        return len(self._lab)

sourceDateset = MyCustomDataset()
sourceDataset_loader=data.DataLoader(sourceDateset,shuffle=True, batch_size=10)
target_dataset= MyCustomDataset2()
target_dataset_loader=data.DataLoader(target_dataset,shuffle=True, batch_size=10)

def train():
    domainModel=CYModel()
    domainModel.cuda()
    lr=0.001
    lamda=1
    domain_loss = nn.CrossEntropyLoss()
    task_loss = nn.CrossEntropyLoss()
    cycle_loss=nn.L1Loss()

    type=["generator"]

    epochs=50
    epochsGenerator=50
    epochsDist=50

    optimizer1 = torch.optim.Adam([{'params':domainModel.tranferLay1.parameters()},{'params':domainModel.tranferLay2.parameters()},
                                   {'params':domainModel.tranferLay_T1.parameters()},{'params':domainModel.tranferLay_T2.parameters()}],lr=lr)
    optimizer2=torch.optim.Adam(domainModel.discriminatoX.parameters(),lr=lr)
    optimizer3=torch.optim.Adam(domainModel.discriminatoY.parameters(),lr=lr)

    domainModel.xvectorSource.state_dict().values()

    for epochDist in range(10):
        for i, (dataSource, dataTarget) in enumerate(zip(sourceDataset_loader, target_dataset_loader), 0):
            inputs, labels = dataSource
            inputs = inputs.type(torch.FloatTensor)
            inputt, labelt = dataTarget
            inputt = inputt.type(torch.FloatTensor)
            inputs, labels = inputs.cuda(), labels.cuda()
            inputt, labelt = inputt.cuda(), labelt.cuda()
            if(labelt.shape[0]<10):
                break
            target2sourceF, target2sourceX, source2targetX, source2targetF, sourceCycleR, tagetCycleR, sourceCycleX, targetCycleX, \
            xvectorx, xvectory, disx, disy, disx2x, disx2y, disy2x, disy2y, x_F, y_F = domainModel(inputs, inputt, 1)
            # lossDiscA = domain_loss(disx, torch.ones(len(disx)).long().cuda()) + domain_loss(disy2x, torch.zeros(
            #     len(disy2x)).long().cuda())
            # lossDiscB = domain_loss(disy, torch.ones(len(disy)).long().cuda()) + domain_loss(disx2y, torch.zeros(
            #     len(disx2y)).long().cuda())

            target2sourceF2, target2sourceX2, source2targetX2, source2targetF2, sourceCycleR2, tagetCycleR2, sourceCycleX2, targetCycleX2, \
            xvectorx2, xvectory2, disx2, disy2, disx2x2, disx2y2, disy2x2, disy2y2, x_F2, y_F2 = domainModel(inputt, inputs, 1)

            lossDiscA =domain_loss(disx, torch.ones(len(disx)).long().cuda()) + domain_loss(disx2, torch.zeros(
                len(disx2)).long().cuda())

            lossDiscB = domain_loss(disx, torch.ones(len(disy)).long().cuda()) + domain_loss(disy2, torch.zeros(
                len(disx2)).long().cuda())

            print("disx:",torch.sum(torch.max(disx, 1)[1]))
            # print("disy2x:", torch.sum(torch.max(disy2x, 1)[1]))
            # print("disy:", torch.sum(torch.max(disy, 1)[1]))
            # print("disx2y:", torch.sum(torch.max(disx2y, 1)[1]))
            print("disx2:",torch.sum(torch.max(disx2, 1)[1]))
            # print("disy2x2:", torch.sum(torch.max(disy2x2, 1)[1]))
            # print("disy2:", torch.sum(torch.max(disy2, 1)[1]))
            # print("disx2y2:", torch.sum(torch.max(disx2y2, 1)[1]))
            # print((torch.sum(torch.max(disy, 1)[1])+ torch.sum(torch.ones(len(disx2y)).long().cuda()-
            #     torch.max(disx2y, 1)[1] - torch.zeros(len(disx2y)).long().cuda())).float() / (len(disx)+len(disy2x)))
            # elif "distimator" in type:
            optimizer2.zero_grad()
            # optimizer3.zero_grad()
            lossDiscA.backward()
            # lossDiscB.backward()
            optimizer2.step()
            # optimizer3.step()


    for epoch in range(epochs):
        acc = 0
        for epochGenerator in range(epochsGenerator):
            times=0
            for i, (dataSource, dataTarget) in enumerate(zip(sourceDataset_loader, target_dataset_loader), 0):
                if times==50:
                    break
                else:
                    times+=1
                print("haha")
                inputs, labels = dataSource
                inputs = inputs.type(torch.FloatTensor)
                inputt, labelt = dataTarget
                inputt = inputt.type(torch.FloatTensor)
                inputs, labels = inputs.cuda(), labels.cuda()
                inputt, labelt = inputt.cuda(), labelt.cuda()
                target2sourceF, target2sourceX, source2targetX, source2targetF, sourceCycleR, tagetCycleR, sourceCycleX, targetCycleX, \
                xvectorx, xvectory, disx, disy, disx2x, disx2y, disy2x, disy2y, x_F, y_F = domainModel(inputs, inputt, 1)
                lossDomainS = domain_loss(disy2x, torch.ones(len(disy2x)).long().cuda())
                lossDiscA = domain_loss(disx, torch.ones(len(disx)).long().cuda()) + domain_loss(disy2x, torch.zeros(
                    len(disy2x)).long().cuda())
                lossDomainT = domain_loss(disx2y, torch.ones(len(disx2y)).long().cuda())
                lossDiscB = domain_loss(disy, torch.ones(len(disy)).long().cuda()) + domain_loss(disx2y, torch.zeros(
                    len(disx2y)).long().cuda())
                lossCycleA = cycle_loss(xvectorx, sourceCycleX)
                lossCycleB = cycle_loss(xvectory, targetCycleX)
                # lossClassficss=task_loss(x_F,label)
                # lossClassficst = task_loss(source2targetF, label)
                lossAllGeneratpr = lossDomainS + lossDomainT + lamda * lossCycleA + lossCycleB
                # losstarget=lossClassficss+lossClassficst

                #输出
                print((torch.sum(torch.max(disx, 1)[1] - torch.ones(len(disx))).long() + torch.sum(
                    torch.max(disy2x, 1)[1] - torch.zeros(len(disy2x))).long()) / (len(disx) + len(disy2x)))
                print((torch.sum(torch.max(disy, 1)[1] - torch.ones(len(disy))).long() + torch.sum(
                    torch.max(disx2y, 1)[1] - torch.zeros(len(disx2y))).long()) / (len(disx) + len(disy2x)))

                # if "generator" in type:
                optimizer1.zero_grad()
                lossAllGeneratpr.backward()
                optimizer1.step()


        for epochDist in range(epochsDist):
            times = 0
            for i, (dataSource, dataTarget) in enumerate(zip(sourceDataset_loader, target_dataset_loader), 0):
                if times==50:
                    break
                else:
                    times+=1
                inputs, labels = dataSource
                inputs = inputs.type(torch.FloatTensor)
                inputt, labelt = dataTarget
                inputt = inputt.type(torch.FloatTensor)
                inputs, labels = inputs.cuda(), labels.cuda()
                inputt, labelt = inputt.cuda(), labelt.cuda()
                target2sourceF, target2sourceX, source2targetX, source2targetF, sourceCycleR, tagetCycleR, sourceCycleX, targetCycleX, \
                xvectorx, xvectory, disx, disy, disx2x, disx2y, disy2x, disy2y, x_F, y_F = domainModel(inputs, inputt, 1)
                lossDomainS = domain_loss(disy2x, torch.ones(len(disy2x)).long().cuda())
                lossDiscA = domain_loss(disx, torch.ones(len(disx)).long().cuda()) + domain_loss(disy2x, torch.zeros(
                    len(disy2x)).long().cuda())
                lossDomainT = domain_loss(disx2y, torch.ones(len(disx2y)).long().cuda())
                lossDiscB = domain_loss(disy, torch.ones(len(disy)).long().cuda()) + domain_loss(disx2y, torch.zeros(
                    len(disx2y)).long().cuda())
                lossCycleA = cycle_loss(xvectorx, sourceCycleX)
                lossCycleB = cycle_loss(xvectory, targetCycleX)
                # lossClassficss=task_loss(x_F,label)
                # lossClassficst = task_loss(source2targetF, label)
                lossAllGeneratpr = lossDomainS + lossDomainT + lamda * lossCycleA + lossCycleB
            # losstarget=lossClassficss+lossClassficst


                #输出
                print((torch.sum(torch.max(disx, 1)[1] - torch.ones(len(disx))).long() + torch.sum(
                    torch.max(disy2x, 1)[1] - torch.zeros(len(disy2x))).long()) / (len(disx) + len(disy2x)))
                print((torch.sum(torch.max(disy, 1)[1] - torch.ones(len(disy))).long() + torch.sum(
                    torch.max(disx2y, 1)[1] - torch.zeros(len(disx2y))).long()) / (len(disx) + len(disy2x)))
            # elif "distimator" in type:
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                lossDiscA.backward()
                lossDiscB.backward()
                optimizer2.step()
                optimizer3.step()
        torch.save(domainModel.state_dict(), '../model/domainmodel{}'.format(epoch))



    # target2sourceF,target2sourceX,source2targetX,source2targetF,sourceCycleR,tagetCycleR,sourceCycleX,targetCycleX, \
    # xvectorx, xvectory, disx, disy, disx2x, disx2y, disy2x, disy2y,x_F,y_F=domainModel(z,z,1)
    # lossDomainS=domain_loss(disy2x,torch.ones(len(disy2x)).long().cuda())
    # lossDiscA=domain_loss(disx,torch.ones(len(disx)).long().cuda())+domain_loss(disy2x,torch.zeros(len(disy2x)).long().cuda())
    # lossDomainT=domain_loss(disx2y,torch.ones(len(disx2y)).long().cuda())
    # lossDiscB=domain_loss(disy,torch.ones(len(disy)).long().cuda())+domain_loss(disx2y,torch.zeros(len(disx2y)).long().cuda())
    # lossCycleA=cycle_loss(xvectorx,sourceCycleX)
    # lossCycleB=cycle_loss(xvectory,targetCycleX)
    # # lossClassficss=task_loss(x_F,label)
    # # lossClassficst = task_loss(source2targetF, label)
    # lossAllGeneratpr=lossDomainS+lossDomainT+lamda*lossCycleA+lossCycleB
    # # losstarget=lossClassficss+lossClassficst
    #
    # if "generator" in type:
    #     optimizer2.zero_grad()
    #     optimizer3.zero_grad()
    #     lossAllGeneratpr.backward()
    #     optimizer2.step()
    #     optimizer3.step()
    # elif "distimator" in type:
    #     optimizer1.zero_grad()
    #     lossDiscA.backward()
    #     lossDiscB.backward()
    #     optimizer1.step()


    # lossDomainT2S=domain_loss(targ)

    # print(feature,t)

if __name__ == '__main__':
    # conv1 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3)
    # input = torch.randn(50, 35, 300)
    # # batch_size x max_sent_len x embedding_size -> batch_size x embedding_size x max_sent_len
    # input = input.permute(0, 2, 1)
    # print("input:", input.size())
    # output = conv1(input)
    # print("output:", output.size())
    # domain1=BYModel()
    # domain1.cuda()
    # domain1.eval()
    # domain2 = BYModel()
    # domain2.cuda()
    # domain2.eval()
    # z = np.load("/media/dream/新加卷/ashell1Data/dataMfcc/voxenroll/s0077-s0077wk0002.npy")
    # z = torch.from_numpy(z)
    # z = z.reshape(1, z.shape[0], z.shape[1])
    # z = z.type(torch.FloatTensor)
    # z = z.cuda()
    # # print(type(z))
    # domainall=CYModel(domain1,domain2)
    # domainall.cuda()
    # domainall.eval()
    # print(domainall(z,z,1))
    # feature,_,_,_,_,_,_,_= domainall(z,z,1)
    train()
