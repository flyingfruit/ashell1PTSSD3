import os
import random

def tries(rate=0.2):
    dir = '/media/dream/新加卷/pldaTry/forTrial/trials'
    a = '/media/dream/新加卷/ashell1Data/dataVadMfcc/enroll'
    b = '/media/dream/新加卷/ashell1Data/dataVadMfcc/test'
    c = '/media/dream/新加卷/ashell1Data/dataVadMfcc/train'

    speakers1 = os.listdir(a)
    speakers2 = os.listdir(b)

    with open(dir + '/' + 'num_ut', 'w') as f:
        i = 0
        while i < 200:
            num1 = random.randint(0, len(speakers1) - 1)
            num2 = random.randint(0, len(speakers2) - 1)
            # one=speakers1[num1][:-4]
            one = speakers1[num1][:5]
            two = speakers2[num2][:-4]
            if one[1:5] != two[1:5] and random.random() < rate:
                continue
            f.write(one + ' ' + two + ' ' + ('target' if one[1:5] == two[1:5] else 'nontarget') + '\n')
            i = i + 1


def prepare(rate=0.2):
    # tries(rate)
    os.system('cd /media/dream/新加卷/pldaTry/ && /media/dream/新加卷/pldaTry/enroll2.sh')


if __name__ == '__main__':
    prepare(0.9)
    scoreDir='/media/dream/新加卷/pldaTry/score/scoress'
    trialsDir='/media/dream/新加卷/pldaTry/forTrial/trials'
    saveDIR='/media/dream/新加卷/pldaTry/forTrial/scorefor'
    with open(saveDIR+'/'+'scoreforeer','w') as score_t:
        with open(scoreDir+'/'+'scores','r') as f2:
            line=f2.readline()
            while(line):
                # [m,n,score]=f2.readline().split()
                [m, n, score] = line.split()
                if m==n[:5]:
                    target='target'
                else:
                    target='nontarget'
                line=f2.readline()
                score_t.write(score+' '+target+'\n')
    os.system(". /media/dream/新加卷/pldaTry/path.sh && compute-eer %s"%(saveDIR+'/'+'scoreforeer'))
