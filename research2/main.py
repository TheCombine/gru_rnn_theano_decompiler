from DataEngine import *
from Vocabulary import *
from TheanoGRU4_1 import *

import sys
import os
import time
import numpy as np
from datetime import datetime
from theano import function, config, shared, tensor

#theano.config.optimizer='None'
#theano.config.exception_verbosity='high'
theano.config.device='cuda'
theano.config.floatX='float32'

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = np.random.RandomState(22)
x = shared(np.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if np.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')


t_start = None

def PrintTime(message, t_start):
    if t_start != None:
        print 'Complete. Took %s s' % (time.time() - t_start)
    print '%s: %s' % (message, time.ctime())
    return time.time()

t_start = PrintTime('Loading vocab', t_start)
voc = LoadVocabulary()
if voc == None:
    voc = Vocabulary()

#print 'Expected Loss for random predictions: %f' % ((np.log(len(voc.token_to_index_obj)) + np.log(len(voc.token_to_index_src))) / 2)

t_start = PrintTime('building GRUDecompiler', t_start)
decompiler = LoadModel(word_dim_obj=len(voc.token_to_index_obj),
                        hidden_dim_obj=256, 
                        learning_rate_obj=0.05, 
                        word_dim_src=len(voc.token_to_index_src),
                        hidden_dim_src=256, 
                        learning_rate_src=0.05)

t_start = PrintTime('Loading data', t_start)
de = DataEngine()

SeqDataToLoad = [o1train, o2train, o3train]
for DataToLoad in SeqDataToLoad:
    SeqSeqTokenObj, SeqSeqTokenSrc, SeqStringAli = de.LoadData(DataToLoad)

    #t_start = PrintTime('converting tokens to indices', t_start)
    SeqSeqIndexObj = voc.ToIndicesObj(SeqSeqTokenObj)
    SeqSeqIndexSrc = voc.ToIndicesSrc(SeqSeqTokenSrc)

    #t_start = PrintTime('cloning sequences of indices', t_start)
    SeqSeqIndexObjClone = voc.CloneSeqSeq(SeqSeqIndexObj)
    SeqSeqIndexSrcClone = voc.CloneSeqSeq(SeqSeqIndexSrc)

    #longest = 0
    #t_start = PrintTime('appending special tokens', t_start)
    for SeqIndexObj in SeqSeqIndexObj:
        #if len(SeqIndexObj) > longest:
        #    longest = len(SeqIndexObj)
        SeqIndexObj.insert(0, 0)
    for SeqIndexSrc in SeqSeqIndexSrc:
        #if len(SeqIndexSrc) > longest:
        #    longest = len(SeqIndexSrc)
        SeqIndexSrc.insert(0, 0)
    for SeqIndexObjClone in SeqSeqIndexObjClone:
        #if len(SeqIndexObjClone) > longest:
        #    longest = len(SeqIndexObjClone)
        SeqIndexObjClone.append(1)
    for SeqIndexSrcClone in SeqSeqIndexSrcClone:
        #if len(SeqIndexSrcClone) > longest:
        #    longest = len(SeqIndexSrcClone)
        SeqIndexSrcClone.append(1)
    #print 'Longest sequence: ', longest

    #t_start = PrintTime('Starting training', t_start)
    decompiler.train(SeqSeqIndexObj, SeqSeqIndexObjClone, SeqSeqIndexSrc, SeqSeqIndexSrcClone, 1, 256)

    '''
    t_start = PrintTime('Decompiling', t_start)
    print 'Excepted shape: ', np.shape(SeqSeqIndexSrcClone[0])
    print 'Excepted: ', SeqSeqIndexSrcClone[0]
    w_pred = decompiler.decompile(SeqSeqIndexObj[0], SeqSeqIndexObjClone[0])
    print 'Prediction: ', w_pred
    '''

PrintTime('Ending Program', t_start)

