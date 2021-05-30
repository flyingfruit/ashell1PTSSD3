import os
import numpy as np


train_path="v1/mfcc/train"
test_path="v1/mfcc/test"
train_txt_path="/home/dream/PycharmProjects/cantsuccess/v1/mfcc_txt/train_txt"
text_txt_path="/home/dream/PycharmProjects/cantsuccess/v1/mfcc_txt/test_txt"
mfcc_npy_train_path="/home/dream/PycharmProjects/cantsuccess/mfcc_npy/train/"
mfcc_npy_test_path="/home/dream/PycharmProjects/cantsuccess/mfcc_npy/test/"

def scp2txt():
    raws = os.listdir(train_path)
    for raw in raws:
        if "ark" in raw:
            result=os.system(". ./v1/path.sh && copy-feats ark:%s ark,t:%s.txt"%("/home\
            /dream/PycharmProjects/cantsuccess/v1/mfcc/train/"+raw,text_txt_path+'/'+os.path.splitext(raw)[0]))

def txt2npy():
    raws = os.listdir(train_txt_path)
    for raw in raws:
        fun3(train_txt_path+'/'+raw)


def fun3(f):
    mfcc=open(f)
    i=1
    flag=0
    line=mfcc.readline()
    while line:
        print(line)
        # line=line.strip().split(' ')
        # split_data = [item for item in line]
        # if '[' in split_data[-1]:
        #     name=line[0]
        #     line=mfcc.readline()
        #     line = line.strip().split(' ')
        #     split_data = [item for item in line]
        #     features_item = np.array(split_data).astype('float')
        #     mfcc_feat_npy=features_item[np.newaxis,:]
        # elif ']' in split_data[-1]:
        #     features_item = np.array(split_data[0:-1]).astype('float')
        #     mfcc_feat_npy=np.append(mfcc_feat_npy,features_item[np.newaxis,:],axis=0)
        #     np.save(mfcc_npy_train_path+name, mfcc_feat_npy)
        # else:
        #     features_item = np.array(split_data).astype('float')
        #     mfcc_feat_npy = np.append(mfcc_feat_npy, features_item[np.newaxis,:],axis=0)

        line=mfcc.readline()
    pass


if __name__ == '__main__':
    fun3('xvector.txt')
    # txt2npy()
