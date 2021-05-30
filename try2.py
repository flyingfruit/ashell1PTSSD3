import random
import sys

# python3 try2.py /media/dream/新加卷/ashell1Data/dataxvectordeal/voxenroll /media/dream/新加卷/ashell1Data/dataxvectordeal/voxtest /media/dream/新加卷/ashell1Data/dataxvectordeal/plda_try

if __name__ == '__main__':
    t=sys.argv
    with open('{}/num_utts.ark'.format(t[1],'r')) as f:
        spksE=[]
        uttsE=[]
        for line in f.readlines():
            [a,b]=line.split()
            spksE.append(b)
            uttsE.append(a)
        spksE=set(spksE)
    with open('{}/utt2spk'.format(t[2]),'r') as f:
        spksT=[]
        uttsT=[]
        for line in f.readlines():
            [a,b]=line.split()
            spksT.append(b)
            uttsT.append(a)
        spksT=set(spksT)

    with open('{}/num_ut'.format(t[3]), 'w') as f:
        for i in range(20000):
            for uttE in uttsE:
                num=random.randint(0,len(uttsT)-1)
                num2 = random.randint(0, len(uttsT) - 1)
                if uttE==uttsT[num][0:7]:
                    # print("发现")
                    target='target'
                    # f.write(str(uttE) + ' ' + uttsT[num] + ' ' + target + '\n')
                    # exit()
                else:
                    if num2<(len(uttsT)-1)/2:
                        continue
                    target='nontarget'

                f.write(str(uttE)+' '+uttsT[num]+' '+target+'\n')
            # print(line)

    # with open('num_ut_p', 'w') as f:
    #     for spk in spks:
    #         for i in range(20):
    #             num=random.randint(0,len(utts)-1)
    #             # print(utts[num])
    #             if spk==utts[num][0:5]:
    #                 target='target'
    #             else:
    #                 target='nontarget'
    #                 # print(1)
    #             f.write(str(spk)+' '+utts[num]+' '+target+'\n')
        # print(line)
