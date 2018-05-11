import numpy as np
import theano as theano
import theano.tensor as T
# 05/06 kaip tinkls veikia bet index out of bounds uzkodavime (paduotas src vocab len o indexuoja uzkoduojant kaip obj vocab)
class TheanoGRU(object):

    def __init__(self, word_dim, hidden_dim, bptt_truncate):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # learning rate
        self.learning_rate = theano.shared(name='learning_rate', value=0)

        # input word weights
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))

        # GRU weights (Uz, Ur, Uh)
        #U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (3, hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))

        # GRU weights (Wz, Wr, Wh)
        #W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
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

        self.__theano_build__()

    def __theano_build__(self):

        # Encoding
        def EncoderForwardProp(IndexObj, s_tm1, E, U, W, b):
            xt = E[:,IndexObj]
            ###
            z = T.nnet.hard_sigmoid(T.dot(U[0], xt) + T.dot(W[0], s_tm1) + b[0])
            r = T.nnet.hard_sigmoid(T.dot(U[1], xt) + T.dot(W[1], s_tm1) + b[1])
            h = T.tanh(T.dot(U[2], xt) + T.dot(W[2], s_tm1 * r) + b[2])
            #z = T.nnet.hard_sigmoid(U[0].dot(xt) + W[0].dot(s_tm1) + b[0])
            #r = T.nnet.hard_sigmoid(U[1].dot(xt) + W[1].dot(s_tm1) + b[1])
            #h = T.tanh(U[2].dot(xt) + W[2].dot(s_tm1 * r) + b[2])
            ###
            st = (T.ones_like(z) - z) * h + z * s_tm1
            return st

        SeqIndexObj = T.ivector('SeqIndexObj')

        s_enc, updates_enc = theano.scan(
            EncoderForwardProp,
            sequences=SeqIndexObj,
            # truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[self.E, self.U, self.W, self.b]
            )

        # Decoding
        def DecoderForwardProp(s_tm1, o_tm1, E, U, W, b, V, c):
            xt = E[:,T.argmax(o_tm1)]
            ###
            z = T.nnet.hard_sigmoid(T.dot(U[0], xt) + T.dot(W[0], s_tm1) + b[0])
            r = T.nnet.hard_sigmoid(T.dot(U[1], xt) + T.dot(W[1], s_tm1) + b[1])
            h = T.tanh(T.dot(U[2], xt) + T.dot(W[2], s_tm1 * r) + b[2])
            #z = T.nnet.hard_sigmoid(U[0].dot(xt) + W[0].dot(s_tm1) + b[0])
            #r = T.nnet.hard_sigmoid(U[1].dot(xt) + W[1].dot(s_tm1) + b[1])
            #h = T.tanh(U[2].dot(xt) + W[2].dot(s_tm1 * r) + b[2])
            ###
            st = (T.ones_like(z) - z) * h + z * s_tm1
            ot = T.nnet.softmax(T.dot(V, st) + c)[0]
            return [st, ot]

        MaxSrcLen = T.iscalar('MaxSrcLen')

        [s_dec, o_dec], updates_dec = theano.scan(
            DecoderForwardProp,
            # truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=s_enc[-1]),
                          dict(initial=T.zeros(self.word_dim))],
            non_sequences=[self.E, self.U, self.W, self.b, self.V, self.c],
            n_steps=MaxSrcLen
            )

        #Cost
        SeqIndexSrc = T.ivector('SeqIndexSrc')

        cost = T.sum(T.nnet.categorical_crossentropy(o_dec, SeqIndexSrc)) #MaxSrcLen, SeqIndexObj, SeqIndexSrc
        
        # Gradients
        dE = T.grad(cost, self.E) #MaxSrcLen, SeqIndexObj, SeqIndexSrc
        dU = T.grad(cost, self.U) #MaxSrcLen, SeqIndexObj, SeqIndexSrc
        dW = T.grad(cost, self.W) #MaxSrcLen, SeqIndexObj, SeqIndexSrc
        db = T.grad(cost, self.b) #MaxSrcLen, SeqIndexObj, SeqIndexSrc
        dV = T.grad(cost, self.V) #MaxSrcLen, SeqIndexObj, SeqIndexSrc
        dc = T.grad(cost, self.c) #MaxSrcLen, SeqIndexObj, SeqIndexSrc
        
        # Prediction
        prediction = T.argmax(o_dec, axis=1) #SeqIndexObj, SeqIndexSrc

        # Assign functions
        self.predict = theano.function([MaxSrcLen, SeqIndexObj], prediction)

        self.error = theano.function([MaxSrcLen, SeqIndexObj, SeqIndexSrc], cost)

        #self.bptt = theano.function([MaxSrcLen, SeqIndexObj, SeqIndexSrc], [dE, dU, dW, db, dV, dc])

        self.sgd_step = theano.function(
            [MaxSrcLen, SeqIndexObj, SeqIndexSrc],
            [], 
            updates=[(self.E, self.E - self.learning_rate * dE),
                     (self.U, self.U - self.learning_rate * dU),
                     (self.W, self.W - self.learning_rate * dW),
                     (self.b, self.b - self.learning_rate * db),
                     (self.V, self.V - self.learning_rate * dV),
                     (self.c, self.c - self.learning_rate * dc)
                    ])

        self.test1 = theano.function(
            inputs=[SeqIndexObj],
            outputs=[s_enc]
            )

        self.test2 = theano.function(
            inputs=[SeqIndexObj, MaxSrcLen],
            outputs=[s_dec, o_dec]
            )
        
    def calculate_total_loss(self, X, Y):
        return np.sum([self.error(len(y), x, y) for x, y in zip(X, Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)