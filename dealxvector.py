import os
import sys



# 把txt转化后的拷贝到enrolldir和txtdir,c表示是否重写num_ut
def txt2ark(d,enrolldir="/media/dream/新加卷/ashell1Data/dataxvectordeal/1505enroll",testdir="/media/dream/新加卷/ashell1Data/dataxvectordeal/1505test",num='',c=False):
    enrolldir="/media/dream/新加卷/ashell1Data/dataxvectordeal/{}{}".format(d,'enroll{}'.format(num))
    testdir="/media/dream/新加卷/ashell1Data/dataxvectordeal/{}{}".format(d,'test{}'.format(num))
    with open(enrolldir+"/"+"enroll.sh", 'w') as f:
        f.write("""cd {enrolldir}
. /media/dream/新加卷/ashell1TFSSD2/path.sh
copy-vector ark,t:enrollXVector.txt ark,scp:enrollXVector.ark,enrollXVector.scp
cp /media/dream/新加卷/ashell1TFSSD2/{d}/{t}/data/utt2spk utt2spk
cp /media/dream/新加卷/ashell1TFSSD2/{d}/{t}/data/spk2utt spk2utt
ivector-mean ark:spk2utt ark:enrollXVector.ark ark,scp:spkXVector.ark,spkXVector.scp ark,t:num_utts.ark
        """.format(d=d,t='enroll',enrolldir=enrolldir))
    with open(testdir+"/"+"enroll.sh", 'w') as f:
        f.write("""cd {testdir}
. /media/dream/新加卷/ashell1TFSSD2/path.sh
copy-vector ark,t:testXVector.txt ark,scp:testXVector.ark,testXVector.scp
cp /media/dream/新加卷/ashell1TFSSD2/{d}/{t}/data/utt2spk utt2spk
cp /media/dream/新加卷/ashell1TFSSD2/{d}/{t}/data/spk2utt spk2utt
        """.format(d=d,t="test",testdir=testdir))
    # os.system(enrolldir+"/"+"enroll.sh&&"+testdir+'/'+"enroll.sh")
    plda="/media/dream/新加卷/ashell1Data/dataxvectordeal/plda_try"
    if os.path.exists(plda):
        pass
    else:
        os.makedirs(plda)
    plda_train="/media/dream/新加卷/ashell1Data/dataxvectordeal/plda_try"+"/"+"plda_train"
    if os.path.exists(plda_train):
        pass
    else:
        os.makedirs(plda_train)
    if os.path.exists(plda_train+"/"+"log"):
        pass
    else:
        os.makedirs(plda_train+"/"+"log")
    plda_score="/media/dream/新加卷/ashell1Data/dataxvectordeal/plda_try"+"/"+"plda_score"
    if os.path.exists(plda_score):
        pass
    else:
        os.makedirs(plda_score)
    if os.path.exists(plda_score+"/"+"log"):
        pass
    else:
        os.makedirs(plda_score+"/"+"log")
    os.system("utils/fix_data_dir.sh {enrolldir}&&utils/fix_data_dir.sh {testdir}".format(enrolldir=enrolldir,testdir=testdir))
    # print("python3 try2.py {} {} {}".format(enrolldir,testdir,plda))
    if c:
        os.system("python3 try2.py {} {} {}".format(enrolldir,testdir,plda))
    with open(plda+"/"+"enroll.sh", 'w') as f:
        f.write("""cd {plda}
export KALDI_ROOT=/home/dream/PycharmProjects/kaldi
export PATH=/media/dream/新加卷/pldaTry/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
DoDataPre=0
FIXDATA=0
FeatureForMfcc=0
VAD=0
run_xvector=0
EXTRACT=0
pwd=`pwd`
trainPlda=1
score=1
EER=1
train_cmd="run.pl"

if [ $trainPlda -eq 1 ]; then
# Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $pwd/plda_train/log/compute_mean.log \
    ivector-mean ark:{enrolldir}/enrollXVector.ark \
    $pwd/plda_train/mean.vec || exit 1;
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd $pwd/plda_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean ark:{enrolldir}/enrollXVector.ark ark:- |" \
    ark:{enrolldir}/utt2spk $pwd/plda_train/transform.mat || exit 1;
  # Train the PLDA model.
  $train_cmd $pwd/plda_train/log/plda.log \
    ivector-compute-plda ark:{enrolldir}/spk2utt \
    "ark:ivector-subtract-global-mean ark:{enrolldir}/enrollXVector.ark ark:- | transform-vec plda_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $pwd/plda_train/plda || exit 1;
fi

if [ $score -eq 1 ]; then
$train_cmd $pwd/score/log/test_scoring.log \
ivector-plda-scoring --normalize-length=true \
     "ivector-copy-plda --smoothing=0.0 plda_train/plda - |" \
    "ark:ivector-subtract-global-mean $pwd/plda_train/mean.vec ark:{enrolldir}/spkXVector.ark ark:- | transform-vec plda_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $pwd/plda_train/mean.vec ark:{testdir}/testXVector.ark ark:- | transform-vec plda_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat 'num_ut' | cut -d\  --fields=1,2|" score/scores
fi
        """.format(plda=plda,enrolldir=enrolldir,testdir=testdir))

    os.system(plda+"/"+"enroll.sh")
    with open('/media/dream/新加卷/ashell1Data/dataxvectordeal/plda_try/scoreforeer','w') as score_t:
        with open(plda+'/score/scores','r') as f2:
            for line in f2.readlines():
                [m,n,score]=line.split()
                if m[:7]==n[:7]:
                    target="target"
                else:
                    target="nontarget"
                score_t.write(score+' '+target+'\n')
    with open(plda+"/"+"calculateeer.sh", 'w') as f:
        f.write("""EER=1
if [ $EER -eq 1 ]; then
. /media/dream/新加卷/ashell1TFSSD2/path.sh
    compute-eer /media/dream/新加卷/ashell1Data/dataxvectordeal/plda_try/scoreforeer
fi
        """)
    os.system("/media/dream/新加卷/ashell1Data/dataxvectordeal/plda_try/calculateeer.sh")
if __name__ == '__main__':
    txt2ark('vox',c=True)
    # txt2ark('1505',num='2')

