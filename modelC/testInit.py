import torch
from modelC.CModel import CYModel
import numpy as np


if __name__ == '__main__':
    model=torch.load("/media/dream/新加卷/ashell1TFSSD/total_model_new311")
    model_dict=model.state_dict()
    # print(model_dict)
    # print(model)
    model_new=CYModel()
    model_newdict=model_new.state_dict()
    # print("haha")
    # model_newdict.update(model_dict)
    # pretrained_dict={k:v for k,v in model_dict.items()}
    # print(pretrained_dict)
    # print(model_newdict)
    # z=model_new.xvectorSource.state_dict().values()
    # model_new.xvectorSource.state_dict().update(model_dict)
    model_new.xvectorSource.FC.state_dict().update(model[6].state_dict())

    z = np.load("/media/dream/新加卷/ashell1Data/dataMfcc/voxenroll/s0077-s0077wk0002.npy")
    z = torch.from_numpy(z)
    z = z.reshape(1, z.shape[0], z.shape[1])
    z = z.type(torch.FloatTensor)
    z = z.cuda()
    # print(type(z))
    model.eval()
    model_new.eval()
    model.cuda()
    model_new.cuda()
    x=model(z)
    target2sourceF, target2sourceX, source2targetX, source2targetF, sourceCycleR, tagetCycleR, sourceCycleX, targetCycleX, \
    xvectorx, xvectory, disx, disy, disx2x, disx2y, disy2x, disy2y, x_F, y_F = model_new(z, z, 1)
    print(x)
    print(x_F)
    print(y_F)
    # print(model_dict,model_newdict)