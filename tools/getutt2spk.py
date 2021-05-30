import os


if __name__ == '__main__':
    # f=open("/media/dream/新加卷/ashell1Data/dataxvectordeal/voxtest/utt2spk")
    m=os.listdir("/media/dream/新加卷/ashell1Data/DANNXvectorC61/enroll")
    # line = f.readline()
    i = 0
    p=open("/media/dream/新加卷/ashell1Data/DANNXvectorC61/enrollTxt/utt2spk",'w')
    for l in m:
        p.write(l[:-4]+' '+l[:7]+'\n')
