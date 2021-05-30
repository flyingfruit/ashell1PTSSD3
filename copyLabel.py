import random
import numpy as np
import os

#创建标签
datadir='/media/dream/新加卷1/datavox/testmfcc'
dataLabel='/media/dream/新加卷/ashell1PTSSD2/dataMFCCtxt/voxtest.txt'
if __name__ == '__main__':
    # with open('dataMfccTxt/trainSubLabel.txt', 'r') as f:
    i=0
    with open(dataLabel,'w') as f:
        file=os.listdir(datadir)
        for file00 in file:
            f.write(str(i)+' '+file00+" "+'0\n')
            i+=1
