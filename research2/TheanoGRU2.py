import numpy as np
import theano as theano
import theano.tensor as T
# 05/08 improvinu TheanoGRU pridedamas obj vocab i modeli kad uzkoduojant neluztu

class GRU(object):
    def __init__(self, word_dim, hidden_dim, bptt_truncate, learning_rate):
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.learning_rate = learning_rate #self.learning_rate = theano.shared(name='learning_rate', value=0.05)

        # word weights
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
        # SeqIndex starts w/ idx0
        SeqIndex = T.ivector('SeqIndex')

        # SeqIndexTruth ends w/ idx1
        SeqIndexTruth = T.ivector('SeqIndexTruth')

        # Max prediction length
        MaxPredLen = T.iscalar('MaxPredLen')

        # Initial prediction state
        InitState = T.dvector('InitState')

        def ForwardProp(IndexEnc, s_tm1, E, U, W, b, V, c):
            xt = E[:,IndexEnc]
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
            outputs_info=[dict(initial=T.zeros(self.hidden_dim)), 
                          None],
            non_sequences=[self.E, self.U, self.W, self.b, self.V, self.c]
            )

        ##### cost function #####
        cost = T.sum(T.nnet.categorical_crossentropy(o, SeqIndexTruth))

        self.cost = theano.function([SeqIndex, SeqIndexTruth], cost)

        ##### train function #####
        dE = T.grad(cost, self.E)
        dU = T.grad(cost, self.U)
        dW = T.grad(cost, self.W)
        db = T.grad(cost, self.b)
        dV = T.grad(cost, self.V)
        dc = T.grad(cost, self.c)

        self.sgd_step = theano.function(
            [SeqIndex, SeqIndexTruth],
            [], 
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
            outputs_info=[dict(initial=InitState),
                          dict(initial=T.zeros((self.hidden_dim)))],
            non_sequences=[self.E, self.U, self.W, self.b, self.V, self.c],
            n_steps=MaxPredLen
            )

        self.predict = theano.function([InitState, MaxPredLen], o_pred)
    
    def calculate_loss(self, SeqSeqIndex, SeqSeqIndexTruth):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(SeqIndex) for SeqIndex in SeqSeqIndex])
        total_loss = np.sum([self.cost(SeqIndex, SeqIndexTruth) for SeqIndex, SeqIndexTruth in zip(SeqSeqIndex, SeqSeqIndexTruth)])
        return total_loss/float(num_words)


class GRUDecompiler(object):

    def __init__(self, word_dim_obj, word_dim_src, hidden_dim_obj, hidden_dim_src, bptt_truncate_obj, bptt_truncate_src, learning_rate_obj, learning_rate_src):
        self.encoder = GRU(word_dim_obj, hidden_dim_obj, bptt_truncate_obj, learning_rate_obj)
        self.decoder = GRU(word_dim_src, hidden_dim_src, bptt_truncate_src, learning_rate_src)

    def train(self, SeqSeqIndexObj, SeqSeqIndexObjTruth, SeqSeqIndexSrc, SeqSeqIndexSrcTruth, nepoch=5, evaluate_loss_every=100):
        num_samples_seen = 0
        last_enc_loss = None
        last_dec_loss = None
        for epoch in xrange(nepoch):
            for SeqIndexObj, SeqIndexObjTruth, SeqIndexSrc, SeqIndexSrcTruth in zip(SeqSeqIndexObj, SeqSeqIndexObjTruth, SeqSeqIndexSrc, SeqSeqIndexSrcTruth):
                if evaluate_loss_every > 0 and num_samples_seen % evaluate_loss_every == 0:
                    print 'Epoch: %s, num_samples_seen: %s' % (epoch, num_samples_seen)
                    print 'Calculating loss.'

                    curr_loss = self.encoder.calculate_loss(SeqSeqIndexObj, SeqSeqIndexObjTruth)
                    print 'Encoder loss: ', curr_loss
                    if last_enc_loss != None and curr_loss > last_enc_loss:
                        self.encoder.learning_rate /= 0.5
                        print 'Lowering encoder learning rate to: ', self.encoder.learning_rate
                    last_enc_loss = curr_loss

                    curr_loss = self.decoder.calculate_loss(SeqSeqIndexSrc, SeqSeqIndexSrcTruth)
                    print 'Decoder loss: ', curr_loss
                    if last_dec_loss != None and curr_loss > last_dec_loss:
                        self.decoder.learning_rate /= 0.5
                        print 'Lowering decoder learning rate to: ', self.encoder.learning_rate
                    last_dec_loss = curr_loss

                self.encoder.sgd_step(SeqIndexObj, SeqIndexObjTruth)
                self.decoder.sgd_step(SeqIndexSrc, SeqIndexSrcTruth)
                num_samples_seen += 1

    def calculate_loss(self, SeqSeqIndexObj, SeqSeqIndexObjTruth, SeqSeqIndexSrc, SeqSeqIndexSrcTruth):
        total_loss = self.encoder.calculate_loss(SeqSeqIndexObj, SeqSeqIndexObjTruth) + self.decoder.calculate_loss(SeqSeqIndexSrc, SeqSeqIndexSrcTruth)
        return total_loss / 2