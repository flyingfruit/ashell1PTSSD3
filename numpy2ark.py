import numpy
import os
if __name__ == '__main__':

    # with open('/media/dream/新加卷/ashell1Data/dataXVector/trainTxt/train.txt') as f:
    #     line=f.readline()
    #     line = line.strip().split(' ')
    #     split_data = [item for item in line]
    #     print(split_data)
    # a=numpy.load('/media/dream/新加卷/ashell1Data/dataXVector/train/S0002-BAC009S0002W0122.npy')
    # print(a.shape)
    # exit(0)
    a=3
    if a==1:
        #存储名字
        sTxt='testXVector.txt'
        #npy地址
        npyDir = '/media/dream/新加卷/ashell1Data/DANNXvectorC70/test'
        #txt存储地址
        txtDir = '/media/dream/新加卷/ashell1Data/DANNXvectorC70/testTxt'
        speakers = os.listdir('/media/dream/新加卷/ashell1Data/DANNXvectorC70/test')
    elif a==2:
        #存储名字
        sTxt='enrollXVector.txt'
        #npy地址
        npyDir = '/media/dream/新加卷/ashell1Data/DANNXvectorC9/enroll'
        #txt存储地址
        txtDir = '/media/dream/新加卷/ashell1Data/DANNXvectorC9/enrollTxt'
        speakers = os.listdir('/media/dream/新加卷/ashell1Data/DANNXvectorC9/enroll')
    elif a==3:
        sTxt = 'enrollVector.txt'
        npyDir = '/media/dream/ACCEL/conversion_eval/enroll'
        txtDir = '/media/dream/ACCEL/conversion_eval/enroll_txt'
        speakers = os.listdir('/media/dream/ACCEL/conversion_eval/enroll')
    # npyDir='/media/dream/新加卷/ashell1Data/dataXVector/train'
    # txtDir='/media/dream/新加卷/ashell1Data/dataXVector/trainTxt'
    # speakers=os.listdir('/media/dream/新加卷/ashell1Data/dataXVector/train')
    with open(txtDir + '/' + sTxt, 'w') as f:
        for speaker in speakers:
            print(numpy.load(npyDir+'/'+speaker).shape)
            xvector=numpy.load(npyDir+'/'+speaker)[0]
            f.write(speaker[:-4]+'  [ ')
            for num in xvector:
                f.write(str(num)+' ')
            f.write(']'+'\n')
    exit(0)
    if a==1:
        os.system('cd /media/dream/新加卷/ashell1Data/dataXVector/trainTxt&&/media/dream/新加卷/ashell1Data/dataXVector/trainTxt/t2a.sh')
    elif a==2:
        os.system('cd /media/dream/新加卷/ashell1Data/dataXVector/enrollTxt&&/media/dream/新加卷/ashell1Data/dataXVector/enrollTxt/t2a.sh&&/media/dream/新加卷/ashell1Data/dataXVector/enrollTxt/getaverage.sh')
    elif a==3:
        os.system(
            'cd /media/dream/新加卷/ashell1Data/dataXVector/testTxt&&/media/dream/新加卷/ashell1Data/dataXVector/testTxt/t2a.sh')
    exit(0)
    with open('temp/xvector.1.txt','r') as f:
        pass
    with open('temp/x.txt','w') as f:
        f.write('s001'+'  '+'[ '+"12.1 "+"13.1 "+']'+'\n')
        f.write('s002' + '  ' + '[ ' + "12.1 " + "13.1 " + ']'+'\n')


def change_xvector(ku,jihe,lb='train'):
    lb='train'
    sTxt='{}XVector.txt'.format(jihe)
    npyDir="/home/aistudio/dataxvector/{}{}".format(ku,jihe)
    speakers = os.listdir('/home/aistudio/dataxvector/{}{}'.format(ku,jihe))
    savedir=npyDir+'Txt'
    if os.path.exists(savedir):
        pass
    else:
        os.makedirs(savedir)
    with open(savedir + '/' + sTxt, 'w') as f:
        for speaker in speakers:
            xvector=numpy.load(npyDir+'/'+speaker)
            f.write(speaker[:-4]+'  [ ')
            for num in xvector:
                f.write(str(num)+' ')
            f.write(']'+'\n')