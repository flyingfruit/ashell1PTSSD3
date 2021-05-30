from torch import nn
from TDNN_gpu import *
from model.discriminator import Discriminator
from modelC.BModel import BYModel
import torch.nn.functional as F


# context = [-2, 2]
# input_dim = 23
# output_dim = 512
# net1 = TDNN(context, input_dim, output_dim, full_context=True)
#
# context = [-2, 1, 2]
# input_dim = 512
# output_dim = 512
# net2 = TDNN(context, input_dim, output_dim, full_context=False)
#
# context = [-3, 1, 3]
# input_dim = 512
# output_dim = 512
# net3 = TDNN(context, input_dim, output_dim, full_context=False)
#
# context = [1]
# input_dim = 512
# output_dim = 512
# net4 = TDNN(context, input_dim, output_dim, full_context=False)
#
# context = [1]
# input_dim = 512
# output_dim = 1500
# net5 = TDNN(context, input_dim, output_dim, full_context=False)
#
# SP = StatsPooling()
# FC = FullyConnected()
# FCRelu = FullyConnectedRelu()
# FC2 = FullyConnected2()
# Final = nn.Linear(512, 340)
# net = nn.Sequential(net1, net2, net3, net4, net5, SP, FC,FCRelu, FC2, Final)
# print(net)



class CYModel(nn.Module):
    def __init__(self):
        super(CYModel,self).__init__()
        self.xvectorSource=BYModel()
        self.xvectorTarget=BYModel()
        self.discriminatoX=Discriminator()
        self.discriminatoY=Discriminator()
        self.tranferLay1=nn.Sequential(nn.Conv1d(512,512,3,padding=1),nn.ReLU())
        self.tranferLay2 = nn.Sequential(nn.Conv1d(512, 512, 3, padding=1), nn.ReLU())
        self.tranferLay_T1 = nn.Sequential(nn.Conv1d(512, 512, 3, padding=1), nn.ReLU())
        self.tranferLay_T2 = nn.Sequential(nn.Conv1d(512, 512, 3, padding=1), nn.ReLU())

    def freezon(self,type):
        if type=="generator":
            # for
            # self.xvectorSource.
            pass

    def forward(self,x,y,label):
        ##
        x_restore=x
        y_restore=y
        x=self.xvectorSource.net1(x)
        x=self.xvectorSource.net2(x)


        y=self.xvectorTarget.net1(y)
        y=self.xvectorTarget.net2(y)


        middx=x
        tx=middx.permute(0,2,1)
        tx=self.tranferLay1(tx)
        tx = self.tranferLay2(tx)
        tx=tx.permute(0,2,1)

        middy=y
        ty=middy.permute(0,2,1)
        ty=self.tranferLay_T1(ty)
        ty = self.tranferLay_T2(ty)
        ty=ty.permute(0, 2, 1)


        #change域
        if label==1:
            y_t=x+tx
        # elif label==2:
            x_t=y+ty
            y=y_t
            x=x_t


        #source xvector
        x=self.xvectorSource.net3(x)
        x=self.xvectorSource.net4(x)
        x=self.xvectorSource.net5(x)
        x=self.xvectorSource.SP(x)
        x=self.xvectorSource.FC(x)
        xvectorS=x
        x=self.xvectorSource.FCRelu(x)
        x=self.xvectorSource.FC2(x)
        x=self.xvectorSource.Final(x)

        #target xvector
        y=self.xvectorTarget.net3(y)
        y=self.xvectorTarget.net4(y)
        y=self.xvectorTarget.net5(y)
        y=self.xvectorTarget.SP(y)
        y=self.xvectorTarget.FC(y)
        xvectorT=y
        y=self.xvectorTarget.FCRelu(y)
        y=self.xvectorTarget.FC2(y)
        y=self.xvectorTarget.Final(y)
        # x=self.features(x)

        #cycle xvector
        tcx=middx.permute(0,2,1)
        tcx=self.tranferLay1(tcx)
        tcx = self.tranferLay2(tcx)
        tcx=self.tranferLay_T1(tcx)
        tcx = self.tranferLay_T2(tcx)
        tcx=tcx.permute(0,2,1)
        tcx=self.xvectorSource.net3(tcx)
        tcx=self.xvectorSource.net4(tcx)
        tcx=self.xvectorSource.net5(tcx)
        tcx=self.xvectorSource.SP(tcx)
        tcx=self.xvectorSource.FC(tcx)
        xvectorSc=tcx   #xvector 源域Cycle
        tcx=self.xvectorSource.FCRelu(tcx)
        tcx=self.xvectorSource.FC2(tcx)
        tcx=self.xvectorSource.Final(tcx)

        tcy = middx.permute(0, 2, 1)
        tcy = self.tranferLay1(tcy)
        tcy = self.tranferLay2(tcy)
        tcy = self.tranferLay_T1(tcy)
        tcy = self.tranferLay_T2(tcy)
        tcy = tcy.permute(0, 2, 1)
        tcy = self.xvectorTarget.net3(tcy)
        tcy = self.xvectorTarget.net4(tcy)
        tcy = self.xvectorTarget.net5(tcy)
        tcy = self.xvectorTarget.SP(tcy)
        tcy = self.xvectorTarget.FC(tcy)
        xvectorTc = tcy   #xvector 目标域Cycle
        tcy = self.xvectorTarget.FCRelu(tcy)
        tcy = self.xvectorTarget.FC2(tcy)
        tcy = self.xvectorTarget.Final(tcy)

        disx2y=self.discriminatoY(xvectorT)
        disy2x=self.discriminatoX(xvectorS)
        disx2x=self.discriminatoX(xvectorSc)
        disy2y=self.discriminatoY(xvectorTc)

        x_F,xvectorx=self.xvectorSource(x_restore)
        y_F,xvectory=self.xvectorTarget(y_restore)
        disx=self.discriminatoX(xvectorx)
        disy=self.discriminatoY(xvectory)

        #原来
        # x:y>x result, xvectorS:y>x xvector, xvectorT:x>y xvector, y:x>y result, tcx:x>y>x result, tcy: y>x>y result
        # xvectorSc:x>y>x xvector, xvectorTc y>x>y xvector, xvectorx:x xvector, xvectory:y xvector
        return x,xvectorS,xvectorT,y,tcx,tcy,xvectorSc,xvectorTc,xvectorx,xvectory,disx,disy,disx2x,disx2y,disy2x,disy2y,x_F,y_F

if __name__ == '__main__':
    CYModel()
