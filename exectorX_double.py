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



# mfcc_train="/media/dream/新加卷/ashell1Data/dataVadMfcc/train"
mfcc_train="/media/dream/新加卷/ashell1Data/dataMfcc/train"
mfcc_enroll="/media/dream/新加卷/ashell1Data/dataVadMfcc/enroll"
mfcc_test="/media/dream/新加卷/ashell1Data/dataVadMfcc/test"
mfcc_voxE='/media/dream/新加卷/ashell1Data/dataMfcc/voxenroll'
mfcc_voxt='/media/dream/新加卷/ashell1Data/dataMfcc/voxtest'

a=1
su='vox/'
frameLength=300
if a==1:
    speakers=os.listdir(mfcc_train)
    txt = 'dataMfccTxt/trainSubLongLabel.txt'
    savedir = '/media/dream/新加卷/ashell1Data/dataXVector/'+su+'train'
elif a==2:
    speakers = os.listdir(mfcc_enroll)
    txt = 'dataMfccTxt/enrollSubLongLabel.txt'
    savedir = '/media/dream/新加卷/ashell1Data/dataXVector/'+su+'enroll'
elif a==3:
    speakers = os.listdir(mfcc_test)
    txt = 'dataMfccTxt/testPldaSubLongLabel.txt'
    savedir = '/media/dream/新加卷/ashell1Data/dataXVector/'+su+'test'
elif a==4:
    speakers = os.listdir(mfcc_voxE)
    txt = 'dataMfccTxt/VoxenrollLabel.txt'
    savedir = '/media/dream/新加卷/ashell1Data/dataXVector/'+su+'voxEnroll'
elif a==5:
    speakers = os.listdir(mfcc_voxt)
    txt = 'dataMfccTxt/VoxtestLabel.txt'
    savedir = '/media/dream/新加卷/ashell1Data/dataXVector/'+su+'voxTest'
train_mfcc=[]
train_label=[]
# txt='dataMfccTxt/enrollLabel.txt'
# savedir='/media/dream/新加卷/ashell1Data/dataXVector/enroll'

class MyCustomDataset(Dataset):
    _lab = []
    def __init__(self):
        # with open('dataMfccTxt/trainLabel.txt', 'r') as f:
        with open(txt, 'r') as f:
            line = f.readline()
            i=0
            while (line):
                line = line.strip().split(' ')
                c = [item for item in line]
                self._lab.append(c)
                line = f.readline()

    def __getitem__(self, item):
        if a==1:
            temp = np.load(mfcc_train + "/" + self._lab[item][1])
        elif a==2:
            temp = np.load(mfcc_enroll + "/" + self._lab[item][1])
        elif a==3:
            temp = np.load(mfcc_test + "/" + self._lab[item][1])
        elif a==4:
            temp = np.load(mfcc_voxE + "/" + self._lab[item][1])
        elif a==5:
            temp = np.load(mfcc_voxt + "/" + self._lab[item][1])
        temp = temp[10:frameLength, :]
        return [temp,self._lab[item][1]]

    def __len__(self):
        return len(self._lab)

train_dataset = MyCustomDataset()
train_dataset_loader=data.DataLoader(train_dataset,shuffle=False, batch_size=256)


def get_xvector():
    # net = torch.load('total_model_trilp1')[:-3]
    # print(net)
    net = torch.load('total_model_new311')[:-3]
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
        for num,key in enumerate(label,0):
            np.save(savedir+'/{}'.format(key), output[num,:])
        print('now:',s)


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
    get_xvector()
    # en_Lda()