import numpy as np
import theano as theano
import theano.tensor as T
import time
# 05/11 improvinu TheanoGRU3 apjungdamas encoder ir decoder

class GRUNetwork(object):
    def __init__(self, word_dim, hidden_dim, learning_rate):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = theano.shared(name='learning_rate', value=learning_rate)

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
            # truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=InitialHiddenState), 
                          None
                          ],
            non_sequences=[self.E, self.U, self.W, self.b, self.V, self.c]
            )

        self.ForwardProp = theano.function([SeqIndex, InitialHiddenState], [s])
        
        ##### cost function #####
        #cost = T.sum(T.nnet.categorical_crossentropy(Outputs, SeqIndexTarget))
        #self.Cost = theano.function([Outputs, SeqIndexTarget], cost)

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

        self.Predict = theano.function([InitialHiddenState, MaxPredLen], [s_pred, o_pred])

    #    self.__theano_build_2__()

    #def __theano_build_2__(self):
    #    InitialHiddenState = T.dvector('InitialHiddenState') #last hidden state of encoder
    #    MaxPredLengthRange = T.ivector('MaxPredLengthRange')

    #    ##### prediction function #####
    #    def Pred(Index, s_tm1, o_tm1, E, U, W, b, V, c):
    #        xt = E[:,T.argmax(o_tm1)]
    #        z = T.nnet.hard_sigmoid(T.dot(U[0], xt) + T.dot(W[0], s_tm1) + b[0])
    #        r = T.nnet.hard_sigmoid(T.dot(U[1], xt) + T.dot(W[1], s_tm1) + b[1])
    #        h = T.tanh(T.dot(U[2], xt) + T.dot(W[2], s_tm1 * r) + b[2])
    #        st = (T.ones_like(z) - z) * h + z * s_tm1
    #        ot = T.nnet.softmax(T.dot(V, st) + c)[0]
    #        return [st, ot]

    #    [s_pred, o_pred], updates_pred = theano.scan(
    #        Pred,
    #        sequences=MaxPredLengthRange,
    #        # truncate_gradient=self.bptt_truncate,
    #        outputs_info=[dict(initial=InitialHiddenState), 
    #                      dict(initial=T.zeros(self.word_dim))
    #                      ],
    #        non_sequences=[self.E, self.U, self.W, self.b, self.V, self.c]
    #        )

    #    self.Predict = theano.function([MaxPredLengthRange, InitialHiddenState], [o_pred])

    def calculate_loss(self, SeqSeqIndex, SeqInitialState, SeqSeqIndexTarget):
        SeqLastState = []
        num_words = np.sum([len(SeqIndex) for SeqIndex in SeqSeqIndex])
        total_cost = 0.
        for SeqIndex, InitialState, SeqIndexTarget in zip(SeqSeqIndex, SeqInitialState, SeqSeqIndexTarget):
            coost, s = self.Coost(SeqIndex, InitialState, SeqIndexTarget)
            total_cost += coost
            SeqLastState.append(s[-1])
        return total_cost/float(num_words), SeqLastState

class GRUDecompiler(object):
    def __init__(self, word_dim_obj, hidden_dim_obj, learning_rate_obj, word_dim_src, hidden_dim_src, learning_rate_src):
        self.encoder = GRUNetwork(word_dim_obj, hidden_dim_obj, learning_rate_obj)
        self.decoder = GRUNetwork(word_dim_src, hidden_dim_src, learning_rate_src)
        print 'word_dim_obj:', word_dim_obj
        print 'word_dim_src:', word_dim_src

    def train(self, SeqSeqIndexObj, SeqSeqIndexTargetObj, SeqSeqIndexSrc, SeqSeqIndexTargetSrc, nepoch=5, evaluate_loss_every=100):
        InitialHiddenStateEnc = np.zeros(self.encoder.hidden_dim)
        num_samples_seen = 0
        last_enc_loss = None
        last_dec_loss = None
        for epoch in xrange(nepoch):
            print 'Epoch %s, current time: %s' % (epoch, time.ctime())
            for SeqIndexObj, SeqIndexTargetObj, SeqIndexSrc, SeqIndexTargetSrc in zip(SeqSeqIndexObj, SeqSeqIndexTargetObj, SeqSeqIndexSrc, SeqSeqIndexTargetSrc):
                if evaluate_loss_every > 0 and num_samples_seen % evaluate_loss_every == 0:
                    print 'Epoch: %s, num_samples_seen: %s' % (epoch, num_samples_seen)
                    print 'Calculating loss.'

                    curr_loss, SeqLastStateEnc = self.encoder.calculate_loss(SeqSeqIndexObj, np.zeros((len(SeqSeqIndexObj), self.encoder.hidden_dim)), SeqSeqIndexTargetObj)
                    print 'Encoder loss: ', curr_loss
                    if last_enc_loss != None and curr_loss > last_enc_loss:
                        self.encoder.learning_rate.set_value(self.encoder.learning_rate.get_value() * 0.9)
                        print 'Lowering encoder learning rate to: ', self.encoder.learning_rate.get_value()
                    last_enc_loss = curr_loss

                    curr_loss, SeqLastStateDec = self.decoder.calculate_loss(SeqSeqIndexSrc, SeqLastStateEnc, SeqSeqIndexTargetSrc)
                    print 'Decoder loss: ', curr_loss
                    if last_dec_loss != None and curr_loss > last_dec_loss:
                        self.decoder.learning_rate.set_value(self.decoder.learning_rate.get_value() * 0.9)
                        print 'Lowering decoder learning rate to: ', self.decoder.learning_rate.get_value()
                    last_dec_loss = curr_loss

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

    #def calculate_loss(self, SeqSeqIndexObj, SeqSeqIndexTargetObj, SeqSeqIndexSrc, SeqSeqIndexTargetSrc):
    #    total_cost = 0.
    #    num_words = 0
    #    #SeqIndex, InitialHiddenState, SeqIndexTarget
    #    for SeqIndexObj, SeqIndexTargetObj, SeqIndexSrc, SeqIndexTargetSrc in zip(SeqSeqIndexObj, SeqSeqIndexTargetObj, SeqSeqIndexSrc, SeqSeqIndexTargetSrc):
    #        coost, s = self.encoder.Coost(SeqIndexObj, np.zeros(self.encoder.hidden_dim), SeqIndexTargetObj)
    #        total_cost += coost
    #        coost, s = self.decoder.Coost(SeqIndexSrc, s[-1], SeqIndexTargetSrc)
    #        total_cost += coost
    #        num_words += len(SeqIndexObj) + len(SeqIndexSrc)
    #    return total_cost / float(num_words)

    def decompile(self, SeqIndexObj, SeqIndexTargetObj):
        lastEncoderState = self.encoder.ForwardProp(SeqIndexObj, np.zeros(self.encoder.hidden_dim))[0][-1] #remove tensor wrapper-bullshit and select last state
        o_pred = self.decoder.Predict(lastEncoderState, 200)[0] #remove theano wrapper
        w_pred = np.argmax(o_pred, axis=1)
        return w_pred