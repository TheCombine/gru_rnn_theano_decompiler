import numpy as np
import theano as theano
import theano.tensor as T
# 05/11 improvinu TheanoGRU3 apjungdamas encoder ir decoder

class GRUDecompiler(object):
    def __init__(self, word_dim_obj, word_dim_src, hidden_dim, learning_rate):
        self.word_dim_obj = word_dim_obj
        self.word_dim_src = word_dim_src
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # token weights
        E_obj = np.random.uniform(-np.sqrt(1./word_dim_obj), np.sqrt(1./word_dim_obj), (hidden_dim, word_dim_obj))
        E_src = np.random.uniform(-np.sqrt(1./word_dim_src), np.sqrt(1./word_dim_src), (hidden_dim, word_dim_src))
        self.E_obj = theano.shared(name='E_obj', value=E_obj.astype(theano.config.floatX))
        self.E_src = theano.shared(name='E_src', value=E_src.astype(theano.config.floatX))

        # GRU weights (Uz, Ur, Uh)
        U_obj = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        U_src = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        self.U_obj = theano.shared(name='U_obj', value=U_obj.astype(theano.config.floatX))
        self.U_src = theano.shared(name='U_src', value=U_src.astype(theano.config.floatX))

        # GRU weights (Wz, Wr, Wh)
        W_obj = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        W_src = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        self.W_obj = theano.shared(name='W_obj', value=W_obj.astype(theano.config.floatX))
        self.W_src = theano.shared(name='W_src', value=W_src.astype(theano.config.floatX))

        # GRU biases (bz, br, bh)
        b_obj = np.zeros((3, hidden_dim))
        b_src = np.zeros((3, hidden_dim))
        self.b_obj = theano.shared(name='b_obj', value=b_obj.astype(theano.config.floatX))
        self.b_src = theano.shared(name='b_src', value=b_src.astype(theano.config.floatX))

        # output weights
        V_obj = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim_obj, hidden_dim))
        V_src = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim_src, hidden_dim))
        self.V_obj = theano.shared(name='V_obj', value=V_obj.astype(theano.config.floatX))
        self.V_src = theano.shared(name='V_src', value=V_src.astype(theano.config.floatX))

        # output bias
        c_obj = np.zeros(word_dim_obj)
        c_src = np.zeros(word_dim_src)
        self.c_obj = theano.shared(name='c_obj', value=c_obj.astype(theano.config.floatX))
        self.c_src = theano.shared(name='c_src', value=c_src.astype(theano.config.floatX))

        # Theano variables
        SeqIndex = T.ivector('SeqIndex') #starts with idx0
        SeqIndexTarget = T.ivector('SeqIndexTarget') #ends with idx1
        InitialHiddenState = T.dscalar('InitialHiddenState') #last hidden state of encoder
        MaxPredLen = T.iscalar('MaxPredLen')

        def ForwardProp(Index, s_tm1, E, U, W, b, V, c):
            xt = E[:,Index]
            z = T.nnet.hard_sigmoid(T.dot(U[0], xt) + T.dot(W[0], s_tm1) + b[0])
            r = T.nnet.hard_sigmoid(T.dot(U[1], xt) + T.dot(W[1], s_tm1) + b[1])
            h = T.tanh(T.dot(U[2], xt) + T.dot(W[2], s_tm1 * r) + b[2])
            st = (T.ones_like(z) - z) * h + z * s_tm1
            ot = T.nnet.softmax(T.dot(V, st) + c)[0]
            return [st, ot]

        [s_enc, o_enc], updates = theano.scan(
            ForwardProp,
            sequences=SeqIndex,
            # truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros(hidden_dim)), 
                          None],
            non_sequences=[E_obj, U_obj, W_obj, b_obj, V_obj, c_obj]
            )

        [s_dec, o_dec], updates = theano.scan(
            ForwardProp,
            sequences=SeqIndex,
            # truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=InitialHiddenState), 
                          None],
            non_sequences=[E_src, U_src, W_src, b_src, V_src, c_src]
            )

        ##### cost functions #####
        cost_enc = T.sum(T.nnet.categorical_crossentropy(o_dec, SeqIndexTarget))
        self.cost_enc = theano.function([SeqIndex, SeqIndexTarget], cost)

        cost_dec = T.sum(T.nnet.categorical_crossentropy(o_dec, SeqIndexTarget))
        self.cost_dec = theano.function([SeqIndex, InitialHiddenState, SeqIndexTarget], cost)

        ##### train functions #####
        dE_obj = T.grad(cost_enc, self.E_obj)
        dU_obj = T.grad(cost_enc, self.U_obj)
        dW_obj = T.grad(cost_enc, self.W_obj)
        db_obj = T.grad(cost_enc, self.b_obj)
        dV_obj = T.grad(cost_enc, self.V_obj)
        dc_obj = T.grad(cost_enc, self.c_obj)

        dE_src = T.grad(cost_dec, self.E_src)
        dU_src = T.grad(cost_dec, self.U_src)
        dW_src = T.grad(cost_dec, self.W_src)
        db_src = T.grad(cost_dec, self.b_src)
        dV_src = T.grad(cost_dec, self.V_src)
        dc_src = T.grad(cost_dec, self.c_src)

        self.sgd_step_enc = theano.function(
            [SeqIndex, SeqIndexTarget],
            [], 
            updates=[(self.E_obj, self.E_obj - self.learning_rate * dE_obj),
                     (self.U_obj, self.U_obj - self.learning_rate * dU_obj),
                     (self.W_obj, self.W_obj - self.learning_rate * dW_obj),
                     (self.b_obj, self.b_obj - self.learning_rate * db_obj),
                     (self.V_obj, self.V_obj - self.learning_rate * dV_obj),
                     (self.c_obj, self.c_obj - self.learning_rate * dc_obj)
                    ])

        self.sgd_step_dec = theano.function(
            [SeqIndex, InitialHiddenState, SeqIndexTarget],
            [], 
            updates=[(self.E_src, self.E_src - self.learning_rate * dE_src),
                     (self.U_src, self.U_src - self.learning_rate * dU_src),
                     (self.W_src, self.W_src - self.learning_rate * dW_src),
                     (self.b_src, self.b_src - self.learning_rate * db_src),
                     (self.V_src, self.V_src - self.learning_rate * dV_src),
                     (self.c_src, self.c_src - self.learning_rate * dc_src)
                    ])

        ##### prediction function #####
        def PredictionForwardProp(s_tm1, o_tm1, E, U, W, b, V, c):
            xt = E[:,T.argmax(o_tm1)]
            z = T.nnet.hard_sigmoid(T.dot(U[0], xt) + T.dot(W[0], s_tm1) + b[0])
            r = T.nnet.hard_sigmoid(T.dot(U[1], xt) + T.dot(W[1], s_tm1) + b[1])
            h = T.tanh(T.dot(U[2], xt) + T.dot(W[2], s_tm1 * r) + b[2])
            st = (T.ones_like(z) - z) * h + z * s_tm1
            ot = T.nnet.softmax(T.dot(V, st) + c)[0]
            return [st, ot]

        [s_pred, o_pred], updates_pred = theano.scan(
            PredictionForwardProp,
            # truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=InitState),
                          dict(initial=T.zeros((self.hidden_dim)))],
            non_sequences=[self.E, self.U, self.W, self.b, self.V, self.c],
            n_steps=MaxPredLen
            )

        self.predict = theano.function([InitState, MaxPredLen], o_pred)
    
    def calculate_loss(self, SeqSeqIndex, InitState, SeqSeqIndexTruth):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(SeqIndex) for SeqIndex in SeqSeqIndex])
        total_loss = np.sum([self.cost(SeqIndex, InitState, SeqIndexTruth) for SeqIndex, SeqIndexTruth in zip(SeqSeqIndex, SeqSeqIndexTruth)])
        return total_loss/float(num_words)


    '''
    TODO:
    calculate_loss encoderyje su initialstate = np.zeros(hidden_dim) gali prasisukti, bet decoderyje reikia masyvu initial state'u
    taigi galbut kurti viena bendra tinkla, o ne isskaidyta i 2 lstm? T.y. kad sutvarkyti cost skaiciavima
    '''