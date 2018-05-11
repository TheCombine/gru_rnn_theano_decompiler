from DataEngine import *
from Vocabulary import *
from TheanoGRU4_1 import *

import sys
import os
import time
import numpy as np
from datetime import datetime

#theano.config.optimizer='None'
#theano.config.exception_verbosity='high'
t_start = None

def PrintTime(message, t_start):
    if t_start != None:
        print 'Complete. Took %s s' % (time.time() - t_start)
    print '%s: %s' % (message, time.ctime())
    return time.time()

x = np.argmax(np.zeros((1000)))
assert x == 0


t_start = PrintTime('Loading vocab', t_start)
voc = LoadVocabulary()
if voc == None:
    voc = Vocabulary()

t_start = PrintTime('Loading data', t_start)
de = DataEngine()

SeqDataToLoad = [o1test]
#SeqDataToLoad = [o1test, o1train, o1valid, o2test, o2train, o2valid, o3test, o3train, o3valid]
for DataToLoad in SeqDataToLoad:
    SeqSeqTokenObj, SeqSeqTokenSrc, SeqStringAli = de.LoadData(DataToLoad)
    voc.FeedObj(SeqSeqTokenObj)
    voc.FeedSrc(SeqSeqTokenSrc)

t_start = PrintTime('cutting sequelnce lengths to 3 samples', t_start)
SeqSeqTokenObj = SeqSeqTokenObj[:3]
SeqSeqTokenSrc = SeqSeqTokenSrc[:3]

t_start = PrintTime('converting tokens to indices', t_start)
SeqSeqIndexObj = voc.ToIndicesObj(SeqSeqTokenObj)
SeqSeqIndexSrc = voc.ToIndicesSrc(SeqSeqTokenSrc)

t_start = PrintTime('cloning sequences of indices', t_start)
SeqSeqIndexObjClone = voc.CloneSeqSeq(SeqSeqIndexObj)
SeqSeqIndexSrcClone = voc.CloneSeqSeq(SeqSeqIndexSrc)

t_start = PrintTime('appending special tokens', t_start)
for SeqIndexObj in SeqSeqIndexObj:
    SeqIndexObj.insert(0, 0)
for SeqIndexSrc in SeqSeqIndexSrc:
    SeqIndexSrc.insert(0, 0)
for SeqIndexObjClone in SeqSeqIndexObjClone:
    SeqIndexObjClone.append(1)
for SeqIndexSrcClone in SeqSeqIndexSrcClone:
    SeqIndexSrcClone.append(1)

t_start = PrintTime('printing examples', t_start)
print 'SeqSeqIndexObj[0]'
print SeqSeqIndexObj[0]
print 'SeqSeqIndexObjClone[0]'
print SeqSeqIndexObjClone[0]
print 'SeqSeqIndexSrc[0]'
print SeqSeqIndexSrc[0]
print 'SeqSeqIndexSrcClone[0]'
print SeqSeqIndexSrcClone[0]

t_start = PrintTime('building GRUDecompiler', t_start)
decompiler = GRUDecompiler(word_dim_obj=len(voc.token_to_index_obj),
                    hidden_dim_obj=100, 
                    learning_rate_obj=0.05, 
                    word_dim_src=len(voc.token_to_index_src),
                    hidden_dim_src=100, 
                    learning_rate_src=0.05)

print 'Expected Loss for random predictions: %f' % ((np.log(len(voc.token_to_index_obj)) + np.log(len(voc.token_to_index_src))) / 2)

t_start = PrintTime('Starting training', t_start)
decompiler.train(SeqSeqIndexObj, SeqSeqIndexObjClone, SeqSeqIndexSrc, SeqSeqIndexSrcClone, 2000, 30)

t_start = PrintTime('Decompiling', t_start)
print 'Excepted shape: ', np.shape(SeqSeqIndexSrcClone[0])
print 'Excepted: ', SeqSeqIndexSrcClone[0]
w_pred = decompiler.decompile(SeqSeqIndexObj[0], SeqSeqIndexObjClone[0])
print 'Prediction: ', w_pred

PrintTime('Ending Program', t_start)
