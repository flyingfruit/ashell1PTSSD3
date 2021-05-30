from torch import nn
from ModelDANN.GRL import GRL
import torch
import os


class DANN2(nn.Module):
    def __init__(self):
        super(DANN2,self).__init__()
        self.num=1
        self.features=nn.Sequential(
            torch.load('../model/total_Xvector{}'.format(self.num))[0][:-1]
        )
        self.featuresRelu=nn.Sequential(
            torch.load('../model/total_Xvector{}'.format(self.num))[0][-1]
        )
        # print(self.features)
        self.task_classifier=nn.Sequential(
            torch.load('../model/total_Xvector{}'.format(self.num))[1]
        )
        # print(self.task_classifier)
        self.domain_classifier=nn.Sequential(
            torch.load('../model/dis{}'.format(self.num))[1]
        )
        # print(self.domain_classifier)
        self.GRL=GRL()
    def forward(self,x,alpha):
        x=self.features(x)

        feature =x
        x=self.featuresRelu(x)
        task_predict=self.task_classifier(x)
        x=GRL.apply(x,alpha)
        domain_predict=self.domain_classifier(x)
        return task_predict,domain_predict,feature

if __name__ == '__main__':
    DANN2()