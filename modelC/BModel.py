from TDNN_gpu import *
import torch


class BYModel(nn.Module):
    def __init__(self):
        super(BYModel,self).__init__()

        self.modelSub=torch.load("/media/dream/新加卷/ashell1TFSSD/total_model_new311")

        # print(self.modelSub)

        self.net1=self.modelSub[0]
        self.net2=self.modelSub[1]
        self.net3=self.modelSub[2]
        self.net4=self.modelSub[3]
        self.net5=self.modelSub[4]
        self.SP=self.modelSub[5]
        self.FC=self.modelSub[6]
        self.FCRelu=self.modelSub[7]
        self.FC2=self.modelSub[8]
        self.Final=self.modelSub[9]

        # context = [-2, 2]
        # input_dim = 23
        # output_dim = 512
        # self.net1 = TDNN(context, input_dim, output_dim, full_context=True)
        #
        # context = [-2, 1, 2]
        # input_dim = 512
        # output_dim = 512
        # self.net2 = TDNN(context, input_dim, output_dim, full_context=False)
        #
        # context = [-3, 1, 3]
        # input_dim = 512
        # output_dim = 512
        # self.net3 = TDNN(context, input_dim, output_dim, full_context=False)
        #
        # context = [1]
        # input_dim = 512
        # output_dim = 512
        # self.net4 = TDNN(context, input_dim, output_dim, full_context=False)
        #
        # context = [1]
        # input_dim = 512
        # output_dim = 1500
        # self.net5 = TDNN(context, input_dim, output_dim, full_context=False)
        #
        # self.SP = StatsPooling()
        # self.FC = FullyConnected()
        # self.FCRelu = FullyConnectedRelu()
        # self.FC2 = FullyConnected2()
        # self.Final = nn.Linear(512, 340)

        # self.net = nn.Sequential(self.net1, self.net2, self.net3, self.net4, self.net5, self.SP, self.FC, self.FCRelu, self.FC2, self.Final)


    def forward(self,x):
        x=self.net1(x)
        x=self.net2(x)
        x=self.net3(x)
        x=self.net4(x)
        x=self.net5(x)
        x=self.SP(x)
        x=self.FC(x)
        xvectorS=x
        x=self.FCRelu(x)
        x=self.FC2(x)
        x=self.Final(x)
        return x,xvectorS

if __name__ == '__main__':
    BYModel()