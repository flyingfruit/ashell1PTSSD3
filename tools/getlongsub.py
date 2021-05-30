dataLabel2="../dataMFCCtxt/trainSubLabelforCycleGan.txt"
dataLabellong="../dataMFCCtxt/trainSubLabelforCycleGanF.txt"
mfcc_train="/media/dream/新加卷/ashell1Data/dataMfcc/train"
mfcc_enroll="/media/dream/新加卷1/subCN-Celeb/enrollMfccnpy"
mfcc_test="/media/dream/新加卷1/subCN-Celeb/testMfccnpy"

import numpy as np
frameLength2=300

with open(dataLabellong,'w') as f2:

    with open(dataLabel2, 'r') as f:
        line = f.readline()
        while (line):
            linetemp=line
            line = line.strip().split(' ')
            c = [item for item in line]
            # if(not os.path.exists(mfcc_enroll + "/" + c[1])):
            #     continue
            temp = np.load(mfcc_train + "/" + c[1])
            print(temp.shape[0], ":", c[1])
            if (temp.shape[0] - 10 < frameLength2):
                print("not")
                line = f.readline()
                continue
            f2.write(linetemp)
            line = f.readline()
