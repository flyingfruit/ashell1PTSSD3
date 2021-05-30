import numpy
import torch
import torch.nn as nn
import xvectorclass
import numpy as np

def train():
    device =torch.device('cuda:0')
    net=xvectorclass.XVectorNet()
    net.cuda()
    optimizer= torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    criterion=nn.CrossEntropyLoss()
    data=np.zeros((64,23,400))
    label=np.zeros((64))
    torchData=torch.from_numpy(data).cuda()
    label=torch.from_numpy(label)
    epoch=1
    print(net)
    for i in range(epoch):
        output=net(torchData).cuda()
        print("haha")
        optimizer.zero_grad()
        loss=criterion(output,0)
        loss.backward()
        optimizer.step()
    print(loss)


    pass

if __name__ == '__main__':
    train()
    # print(torch.cuda.is_available())
    pass