import numpy as np
import theano as theano
import theano.tensor as T

class LSTM(object):

    def __init__(self, word_dim, hidden_dim, bptt_truncate):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # input word weights
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))

        # GRU weights (Uz, Ur, Uh)
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (3, hidden_dim, word_dim))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))

        # GRU weights (Wz, Wr, Wh)
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))

        # GRU biases (bz, br, bh)
        b = np.zeros((3, hidden_dim))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))

        # output word weights
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))

        # output bias
        c = np.zeros(word_dim)
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))

    def softmax(self, x):
        """squashes a vector between 0 and 1"""
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)

    def ForwardPropStates(self, SeqIndexObj, st_1 = None):
        s = np.zeros((len(SeqIndexObj), self.hidden_dim))
        if st_1 == None:
            st_1 = s[-1]

        for i in xrange(len(SeqIndexObj)):
            # Get the word vector
            xt = E[:,SeqIndexObj[i]]

            # GRU Layer
            z = T.nnet.ultra_fast_sigmoid(U[0].dot(xt) + W[0].dot(st_1) + b[0])
            r = T.nnet.ultra_fast_sigmoid(U[1].dot(xt) + W[1].dot(st_1) + b[1])
            h = T.tanh(U[2].dot(xt) + W[2].dot(st_1 * r) + b[2])
            st = (T.ones_like(z) - z) * h + z * st_1

            st_1 = st
            s[i] = st

        return s

    def ForwardPropOutputs(self, SeqTimestepState):
        o = np.zeros((len(SeqTimestepState), self.hidden_dim))
        
        for i in xrange(len(SeqTimestepState)):
            # Theano's softmax returns a matrix with one row, we only need the row
            ot = T.nnet.softmax(V.dot(SeqTimestepState[i]) + c)[0]
            o[i] = ot

        return o
