from TDNN_gpu import *
import numpy as np
import time
import os
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset

mfcc_train="/media/dream/新加卷/ashell1Data/dataMfcc/train"
mfcc_test="/media/dream/新加卷/ashell1TF/dataMfcc/test"
# dataLabel='dataMfccTxt/trainLabel.txt'
dataLabel='dataMfccTxt/trainSubLongLabel.txt'
# frameLength=180
frameLength=300

class MyCustomDataset(Dataset):
    _lab = []

    def __init__(self):
        with open(dataLabel, 'r') as f:
            line = f.readline()
            while (line):
                line = line.strip().split(' ')
                c = [item for item in line]
                self._lab.append(c)
                line = f.readline()

    def __getitem__(self, item):
        temp = np.load(mfcc_train + "/" + self._lab[item][1])
        print(temp.shape)
        print(self._lab[item][1])
        temp = temp[10:frameLength, :]
        return [temp,int(self._lab[item][2])]

    def __len__(self):
        return len(self._lab)



train_mfccs={}
speakers=os.listdir(mfcc_train)

train_dataset = MyCustomDataset()
train_dataset_loader=data.DataLoader(train_dataset,shuffle=False, batch_size=200)



def train():
    context = [-2, 2]
    input_dim = 23
    output_dim = 512
    net1 = TDNN(context, input_dim, output_dim, full_context=True)
    context = [-2, 1, 2]
    input_dim = 512
    output_dim = 512
    net2 = TDNN(context, input_dim, output_dim, full_context=False)

    context = [-3, 1, 3]
    input_dim = 512
    output_dim = 512
    net3 = TDNN(context, input_dim, output_dim, full_context=False)

    context = [1]
    input_dim = 512
    output_dim = 512
    net4 = TDNN(context, input_dim, output_dim, full_context=False)

    context = [1]
    input_dim = 512
    output_dim = 1500
    net5 = TDNN(context, input_dim, output_dim, full_context=False)

    SP = StatsPooling()
    FC = FullyConnected()
    FCRelu = FullyConnectedRelu()
    FC2 = FullyConnected2()
    Final = nn.Linear(512, 340)
    net = nn.Sequential(net1, net2, net3, net4, net5, SP, FC,FCRelu, FC2, Final)
    print(net)
    # net = torch.load('total_model_new311')
    # print(net)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer2 = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    net.cuda()
    epocs = 50
    for j in range(epocs):
        acc=0
        for i, data in enumerate(train_dataset_loader, 0):
            a = time.time()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs=inputs.type(torch.FloatTensor)
            inputs, labels = inputs.cuda(), labels.cuda()
            output = net(inputs).cuda()
            # if(epocs<10):
            #     optimizer2.zero_grad()
            # else:
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            b = time.time()
            print(torch.sum(torch.max(output, 1)[1]- labels == 0))
            acc=acc+torch.sum(torch.max(output, 1)[1] - labels == 0).float()/256
            print("loss: {} step took {}".format(loss, b - a))
            # print("output:\n{} \n labels:\n{}".format(output, labels))
        print(b-a)
        print("acc:",acc/i)
        # torch.save(net, 'total_model_new3{}'.format(j))

if __name__ == '__main__':
    train()