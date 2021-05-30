from TDNN_gpu import *
import numpy as np
import time
import os
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.utils.data.dataset import Dataset



mfcc_train="/media/dream/新加卷/ashell1Data/dataMfcc/train"
mfcc_enroll="/media/dream/新加卷/ashell1Data/dataMfcc/enroll"
mfcc_test="/media/dream/新加卷/ashell1Data/dataMfcc/test"


speakers=os.listdir(mfcc_train)
train_mfcc=[]
train_label=[]
frameLength=300
class MyCustomDataset(Dataset):
    _lab = []
    def __init__(self):
        with open('dataMfccTxt/testCnnSubLongLabel.txt', 'r') as f:
            line = f.readline()
            while (line):
                line = line.strip().split(' ')
                c = [item for item in line]
                self._lab.append(c)
                line = f.readline()

#
    # def __getitem__(self, item):
    #     temp = np.load(mfcc_train + "/" + self._lab[item][1])
    #     temp = temp[10:frameLength, :]
    #     return [temp,self._lab[item][1]]

    # def __getitem__(self, item):
    #     temp = np.load(mfcc_train + "/" + self._lab[item][1])
    #     temp = temp[10:frameLength, :]
    #     return [temp,self._lab[item][1]]

    def __getitem__(self, item):
        temp = np.load(mfcc_train + "/" + self._lab[item][1])
        temp = temp[10:frameLength, :]
        return [temp,int(self._lab[item][2])]

    def __len__(self):
        return len(self._lab)

train_dataset = MyCustomDataset()
train_dataset_loader=data.DataLoader(train_dataset,shuffle=False, batch_size=100)


def get_xvector():
    net = torch.load('total_model_new25')[:-3]
    print(net)
    outputs=0
    labs=0
    for s, data in enumerate(train_dataset_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, label = data
        inputs= inputs.type(torch.FloatTensor)
        inputs=inputs.cuda()
        output = net(inputs)
        output=output.cpu().float().detach().numpy()
        np.save('/media/dream/新加卷/ashell1Data/dataXVector/train/{}'.format(label[0]), output)
        # label=labels.cpu().float().numpy()
        # # output=output[np.newaxis,:]
        # if s==0:
        #     outputs=output
        #     labs=label
        # else:
        #     outputs=np.append(outputs,output,axis=0)
        #     labs=np.append(labs,label)
    # np.save('/media/dream/新加卷/ashell1Data/dataXVector/train/xvector.npy',outputs)
    # np.save('y',labs)


def test_right():
    net = torch.load('total_model_trilp1')
    net.eval()
    i=0
    numCorrect=0
    for s, data in enumerate(train_dataset_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, label = data
        inputs = inputs.type(torch.FloatTensor)
        inputs = inputs.cuda()
        label=label.cuda()
        output = net(inputs).cuda()
        a=output.argmax(dim=1)
        s=(a==label)
        numCorrect=numCorrect+(s.sum())
        print(a)
        print(s)
        print(s.sum())
        i+=100
    print(numCorrect/i)
    print()


def en_Lda():
    LDA=LinearDiscriminantAnalysis(n_components=42)
    xvector = np.load('xvector.npy')
    y=np.load('y.npy')
    xvector_train = np.load('xvector_train.npy')
    y_train = np.load('y_train.npy')
    LDA.fit(xvector_train,y_train)
    x_new=LDA.transform(xvector_train)
    m=LDA.score(xvector,y)
    print(m)
    clases_dict = {}
    for i in range(42):
        clases_dict[i]=[]
    for i, x in enumerate(x_new):
        clases_dict[int(y[i])].append(x)

    fig = plt.figure()
    ax = Axes3D(fig)
    for index in range(42):
        ax.scatter(np.array(clases_dict[index])[:,0],np.array(clases_dict[index])[:,1],np.array(clases_dict[index])[:,2])
    plt.show()


if __name__ == '__main__':
    test_right()
    # get_xvector()
    # en_Lda()