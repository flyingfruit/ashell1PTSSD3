import torch
import torch.nn as nn
import torch.nn.functional as F


class XVectorNet(nn.Module):
    def __init__(self):
        super(XVectorNet, self).__init__()
        self.__inputFeatureDim = 23
        self.__batch = 64
        self.__momentum = 0.1
        self.__numPerson=24
        self.conv1 = nn.Conv1d(self.__inputFeatureDim, 512, 5, stride=1, bias=True).double()
        self.bN1 = nn.BatchNorm1d(512, momentum=self.__momentum).double()
        self.conv2 = nn.Conv1d(512, 512, 5, stride=1, bias=True).double()
        self.bN2 = nn.BatchNorm1d(512, momentum=self.__momentum).double()
        self.conv3 = nn.Conv1d(512, 512, 7, bias=True).double()
        self.bN3 = nn.BatchNorm1d(512, momentum=self.__momentum).double()
        self.conv4 = nn.Conv1d(512, 512, 1, bias=True).double()
        self.bN4 = nn.BatchNorm1d(512, momentum=self.__momentum).double()
        self.conv5 = nn.Conv1d(512, 3 * 512, 1, bias=True).double()
        self.bN5 = nn.BatchNorm1d(3*512, momentum=self.__momentum).double()

        self.fc1=nn.Linear(3*512,512).double()
        self.fc2=nn.Linear(512,512).double()

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.bN1(x)
        x=F.relu(self.conv2(x))
        x = self.bN2(x)
        x = F.relu(self.conv3(x))
        x = self.bN3(x)
        x = F.relu(self.conv4(x))
        x = self.bN4(x)
        x = F.relu(self.conv5(x))
        x = self.bN5(x)
        print(x.size())
        mean=x.mean(dim=1)
        print(mean.size())
        var=x.std(dim=1)
        print(var.size())
        x=torch.cat((mean,var),dim=1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=nn.Linear(512,self.__numPerson)(x)
        return x

