import os
import random

dir='/media/dream/新加卷/pldaTry/forTrial/trials'
a='/media/dream/新加卷/ashell1Data/dataVadMfcc/enroll'
b='/media/dream/新加卷/ashell1Data/dataVadMfcc/test'
c='/media/dream/新加卷/ashell1Data/dataVadMfcc/train'


speakers1=os.listdir(a)
speakers2=os.listdir(b)


with open(dir+'/'+'num_ut', 'w') as f:
    i=0
    while i<200:
        num1=random.randint(0, len(speakers1)-1)
        num2=random.randint(0, len(speakers2)-1)
        # one=speakers1[num1][:-4]
        one=speakers1[num1][:5]
        two=speakers2[num2][:-4]
        if one[1:5] != two[1:5] and random.random()<0.2:
            continue
        f.write(one+' '+ two+' '+('target'if one[1:5]==two[1:5] else 'nontarget')+'\n')
        i=i+1
        # print(line)