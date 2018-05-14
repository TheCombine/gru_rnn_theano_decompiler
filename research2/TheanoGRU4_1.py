import numpy as np
import theano as theano
import theano.tensor as T
import time
from time import strftime, localtime
from os import listdir
import logging
# 05/11 improvinu TheanoGRU3 apjungdamas encoder ir decoder

class GRUNetwork(object):
    def __init__(self, word_dim, hidden_dim, learning_rate):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = theano.shared(name='learning_rate', value=learning_rate)
        self.bptt_truncate = 100

        # token weights
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))

        # GRU weights (Uz, Ur, Uh)
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))

        # GRU weights (Wz, Wr, Wh)
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))

        # GRU biases (bz, br, bh)
        b = np.zeros((3, hidden_dim))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))

        # output weights
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))

        # output bias
        c = np.zeros(word_dim)
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))

        self.__theano_build__()

    def __theano_build__(self):
        # Theano variables
        SeqIndex = T.ivector('SeqIndex') #starts with idx0
        SeqIndexTarget = T.ivector('SeqIndexTarget') #ends with idx1
        InitialHiddenState = T.dvector('InitialHiddenState') #last hidden state of encoder
        Outputs = T.dvector('Outputs') #retrieved from calling ForwardProp
        MaxPredLen = T.iscalar('MaxPredLen')

        ##### forward prop function #####
        def ForwardProp(Index, s_tm1, E, U, W, b, V, c):
            xt = E[:,Index]
            z = T.nnet.hard_sigmoid(T.dot(U[0], xt) + T.dot(W[0], s_tm1) + b[0])
            r = T.nnet.hard_sigmoid(T.dot(U[1], xt) + T.dot(W[1], s_tm1) + b[1])
            h = T.tanh(T.dot(U[2], xt) + T.dot(W[2], s_tm1 * r) + b[2])
            st = (T.ones_like(z) - z) * h + z * s_tm1
            ot = T.nnet.softmax(T.dot(V, st) + c)[0]
            return [st, ot]

        [s, o], updates = theano.scan(
            ForwardProp,
            sequences=SeqIndex,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=InitialHiddenState), 
                          None
                          ],
            non_sequences=[self.E, self.U, self.W, self.b, self.V, self.c]
            )

        self.ForwardProp = theano.function([SeqIndex, InitialHiddenState], [s])
        
        ##### cost function #####
        coost = T.sum(T.nnet.categorical_crossentropy(o, SeqIndexTarget))
        self.Coost = theano.function([SeqIndex, InitialHiddenState, SeqIndexTarget], [coost, s])

        ##### train function #####
        dE = T.grad(coost, self.E)
        dU = T.grad(coost, self.U)
        dW = T.grad(coost, self.W)
        db = T.grad(coost, self.b)
        dV = T.grad(coost, self.V)
        dc = T.grad(coost, self.c)

        self.SGDStep = theano.function(
            [SeqIndex, InitialHiddenState, SeqIndexTarget],
            [s], 
            updates=[(self.E, self.E - self.learning_rate * dE),
                     (self.U, self.U - self.learning_rate * dU),
                     (self.W, self.W - self.learning_rate * dW),
                     (self.b, self.b - self.learning_rate * db),
                     (self.V, self.V - self.learning_rate * dV),
                     (self.c, self.c - self.learning_rate * dc)
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
            outputs_info=[dict(initial=InitialHiddenState),
                          dict(initial=T.zeros(self.word_dim))
                          ],
            non_sequences=[self.E, self.U, self.W, self.b, self.V, self.c],
            n_steps=MaxPredLen
            )

        self.Predict = theano.function([InitialHiddenState, MaxPredLen], [o_pred])

    def calculate_loss(self, SeqSeqIndex, SeqInitialState, SeqSeqIndexTarget):
        SeqLastState = []
        num_words = np.sum([len(SeqIndex) for SeqIndex in SeqSeqIndex])
        total_cost = 0.
        for SeqIndex, InitialState, SeqIndexTarget in zip(SeqSeqIndex, SeqInitialState, SeqSeqIndexTarget):
            coost, s = self.Coost(SeqIndex, InitialState, SeqIndexTarget)
            total_cost += coost
            SeqLastState.append(s[-1])
        return total_cost/float(num_words), SeqLastState


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class GRUDecompiler(object):
    def __init__(self, word_dim_obj, hidden_dim_obj, learning_rate_obj, word_dim_src, hidden_dim_src, learning_rate_src):
        self.encoder = GRUNetwork(word_dim_obj, hidden_dim_obj, learning_rate_obj)
        self.decoder = GRUNetwork(word_dim_src, hidden_dim_src, learning_rate_src)
        logging.basicConfig(filename='logs.txt',level=logging.INFO)
        logging.info("%s | wdo: %s, hdo: %s, lro: %s, wrs: %s, hds: %s, lrs: %s" % (time.ctime(), word_dim_obj, hidden_dim_obj, learning_rate_obj, word_dim_src, hidden_dim_src, learning_rate_src))

    def train(self, SeqSeqIndexObj, SeqSeqIndexTargetObj, SeqSeqIndexSrc, SeqSeqIndexTargetSrc, nepoch=5, evaluate_loss_every=100):
        logging.info("%s | Beginning training. Num training samples: %s, nepoch: %s, evaluate loss every: %s" % (time.ctime(), len(SeqSeqIndexObj), nepoch, evaluate_loss_every))
        InitialHiddenStateEnc = np.zeros(self.encoder.hidden_dim)
        num_samples_seen = 0
        last_enc_loss = None
        last_dec_loss = None
        for epoch in xrange(nepoch):
            logging.info("%s | Beginnig epoch No.%s" % (time.ctime(), epoch))
            for SeqIndexObj, SeqIndexTargetObj, SeqIndexSrc, SeqIndexTargetSrc in zip(SeqSeqIndexObj, SeqSeqIndexTargetObj, SeqSeqIndexSrc, SeqSeqIndexTargetSrc):
                if num_samples_seen % evaluate_loss_every == 0 and num_samples_seen > 0 and evaluate_loss_every > 0:
                    logging.info("%s | Current epoch: %s, samples seen: %s out of %s" % (time.ctime(), epoch, num_samples_seen, len(SeqSeqIndexObj)))
                    samples = np.random.choice(range(len(SeqSeqIndexObj)), 500, replace=False)
                    
                    curr_loss, SeqLastState = self.encoder.calculate_loss([SeqSeqIndexObj[i] for i in samples], np.zeros((len(SeqSeqIndexObj), self.encoder.hidden_dim)), [SeqSeqIndexTargetObj[i] for i in samples])
                    print 'loss is number?', is_number(curr_loss)
                    logging.info("%s | Encoder loss: %s" % (time.ctime(), curr_loss))
                    if last_enc_loss != None and curr_loss > last_enc_loss:
                        self.encoder.learning_rate.set_value(self.encoder.learning_rate.get_value() * 0.9)
                        logging.info("%s | Lowering encoder learning rate to: %s" % (time.ctime(), self.encoder.learning_rate.get_value()))
                    last_enc_loss = curr_loss

                    curr_loss, SeqLastState = self.decoder.calculate_loss([SeqSeqIndexSrc[i] for i in samples], SeqLastState, [SeqSeqIndexTargetSrc[i] for i in samples])
                    logging.info("%s | Decoder loss: %s" % (time.ctime(), curr_loss))
                    if last_dec_loss != None and curr_loss > last_dec_loss:
                        self.decoder.learning_rate.set_value(self.decoder.learning_rate.get_value() * 0.9)
                        logging.info("%s | Lowering decoder learning rate to: %s" % (time.ctime(), self.decoder.learning_rate.get_value()))
                    last_dec_loss = curr_loss

                    self.SaveModel()

                #[SeqIndex, InitialHiddenState, SeqIndexTarget], [s] 
                lastEncoderState = self.encoder.SGDStep(SeqIndexObj, InitialHiddenStateEnc, SeqIndexTargetObj)[0][-1] #remove tensor wrapper-bullshit and select last state
                self.decoder.SGDStep(SeqIndexSrc, lastEncoderState, SeqIndexTargetSrc)

                num_samples_seen += 1

    def calculate_loss(self, SeqSeqIndexObj, SeqSeqIndexTargetObj, SeqSeqIndexSrc, SeqSeqIndexTargetSrc):
        loss, SeqHiddenState = self.encoder.calculate_loss(SeqSeqIndexObj, np.zeros((len(SeqSeqIndexObj), self.encoder.hidden_dim)), SeqSeqIndexTargetObj)
        total_loss = loss
        loss, SeqHiddenState = self.decoder.calculate_loss(SeqSeqIndexSrc, np.zeros((len(SeqSeqIndexSrc), self.decoder.hidden_dim)), SeqSeqIndexTargetSrc)
        total_loss += loss
        return total_loss / 2

    def decompile(self, SeqIndexObj, SeqIndexTargetObj):
        lastEncoderState = self.encoder.ForwardProp(SeqIndexObj, np.zeros(self.encoder.hidden_dim))[0][-1] #remove tensor wrapper-bullshit and select last state
        o_pred = self.decoder.Predict(lastEncoderState, 10)
        w_pred = np.argmax(o_pred, axis=1)
        return w_pred

    def SaveModel(self):
        #outfile = 'savesModel/%s.txt' % strftime('%Y-%m-%d %H;%M;%S', localtime())
        outfile = 'savesModel/model.npz'
        np.savez(outfile,
            E_enc=self.encoder.E.get_value(),
            U_enc=self.encoder.U.get_value(),
            W_enc=self.encoder.W.get_value(),
            b_enc=self.encoder.b.get_value(),
            V_enc=self.encoder.V.get_value(),
            c_enc=self.encoder.c.get_value(),
            lr_enc =self.encoder.learning_rate.get_value(), 
            E_dec=self.decoder.E.get_value(),
            U_dec=self.decoder.U.get_value(),
            W_dec=self.decoder.W.get_value(),
            b_dec=self.decoder.b.get_value(),
            V_dec=self.decoder.V.get_value(),
            c_dec=self.decoder.c.get_value(),
            lr_dec =self.decoder.learning_rate.get_value()
            )
        logging.info("%s | Saved model." % (time.ctime()))
        print 'Saved:'
        print '%s, %s, %s, %s, %s, %s, %s' % (np.shape(self.encoder.E.get_value()), np.shape(self.encoder.U.get_value()), np.shape(self.encoder.W.get_value()), np.shape(self.encoder.b.get_value()), np.shape(self.encoder.W.get_value()), np.shape(self.encoder.c.get_value()), self.encoder.learning_rate.get_value())
        print '%s, %s, %s, %s, %s, %s, %s' % (np.shape(self.decoder.E.get_value()), np.shape(self.decoder.U.get_value()), np.shape(self.decoder.W.get_value()), np.shape(self.decoder.b.get_value()), np.shape(self.decoder.W.get_value()), np.shape(self.decoder.c.get_value()), self.decoder.learning_rate.get_value())

def LoadModel(word_dim_obj, hidden_dim_obj, learning_rate_obj, word_dim_src, hidden_dim_src, learning_rate_src):
    fileNames = listdir('savesModel/')
    if len(fileNames) != 0:
        npzfile = np.load('savesModel/model.npz')
        E_enc, U_enc, W_enc, b_enc, V_enc, c_enc, learning_rate_obj = npzfile["E_enc"], npzfile["U_enc"], npzfile["W_enc"], npzfile["b_enc"], npzfile["V_enc"], npzfile["c_enc"], npzfile["lr_enc"]
        E_dec, U_dec, W_dec, b_dec, V_dec, c_dec, learning_rate_src = npzfile["E_dec"], npzfile["U_dec"], npzfile["W_dec"], npzfile["b_dec"], npzfile["V_dec"], npzfile["c_dec"], npzfile["lr_dec"]
        hidden_dim_obj, word_dim_obj = E_enc.shape[0], E_enc.shape[1]
        hidden_dim_src, word_dim_src = E_dec.shape[0], E_dec.shape[1]

    model = GRUDecompiler(word_dim_obj=word_dim_obj,
                        hidden_dim_obj=hidden_dim_obj, 
                        learning_rate_obj=learning_rate_obj, 
                        word_dim_src=word_dim_src,
                        hidden_dim_src=hidden_dim_src, 
                        learning_rate_src=learning_rate_src)

    if len(fileNames) != 0:
        model.encoder.E.set_value(E_enc)
        model.encoder.U.set_value(U_enc)
        model.encoder.W.set_value(W_enc)
        model.encoder.b.set_value(b_enc)
        model.encoder.V.set_value(V_enc)
        model.encoder.c.set_value(c_enc)
        model.decoder.E.set_value(E_dec)
        model.decoder.U.set_value(U_dec)
        model.decoder.W.set_value(W_dec)
        model.decoder.b.set_value(b_dec)
        model.decoder.V.set_value(V_dec)
        model.decoder.c.set_value(c_dec)
        print 'LOADED model:'
        print '%s, %s, %s, %s, %s, %s, %s' % (np.shape(E_enc), np.shape(U_enc), np.shape(W_enc), np.shape(V_enc), np.shape(b_enc), np.shape(c_enc), learning_rate_obj)
        print '%s, %s, %s, %s, %s, %s, %s' % (np.shape(E_dec), np.shape(U_dec), np.shape(W_dec), np.shape(V_dec), np.shape(b_dec), np.shape(c_dec), learning_rate_src)
    else:
        print 'CREATED new model'
    return model