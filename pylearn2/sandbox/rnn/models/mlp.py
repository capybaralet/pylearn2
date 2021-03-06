import numpy as np
import scipy

from theano import config
from theano import scan
from theano import tensor

from pylearn2.models import mlp
from pylearn2.space import VectorSpace, IndexSpace
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.utils import sharedX
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul

class Recurrent(mlp.Layer):
    """
    A recurrent neural network layer using the hyperbolic
    tangent activation function which only returns its last state
    """
    def __init__(self, dim, layer_name, irange):

        self._rnn_friendly = True
        self.__dict__.update(locals())
        del self.self
        super(Recurrent, self).__init__()

    def set_input_space(self, space):
        # This space expects a VectorSpace sequence
        assert isinstance(space, SequenceSpace)
        assert isinstance(space.space, VectorSpace)

        # Construct the weight matrices and biases
        self.rng = self.mlp.rng
        W_recurrent = self.rng.uniform(-self.irange, self.irange,
                                       (self.dim, self.dim))
        W_recurrent,_,_ = scipy.linalg.svd(W_recurrent)
        
        W_in = self.rng.uniform(-self.irange, self.irange,
                                (space.dim, self.dim))
        

        W_recurrent, W_in = sharedX(0.9 * W_recurrent), sharedX(W_in)
        W_recurrent.name, W_in.name = [self.layer_name + '_' + param for param in ['W_recurrent', 'W_in']]

        b = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b') 

        # Save the parameters and set the output space
        self.params = [W_recurrent, W_in, b]
        self.output_space = VectorSpace(dim=self.dim)
        self.input_space = space

    def fprop(self, state_below):
        # The initial hidden state is just zeros
        h0 = tensor.alloc(np.cast[config.floatX](0),
                          state_below.shape[1], self.dim)
        W_in = self.params[1]

        state_below_in = tensor.dot(state_below, W_in)
        
        def _fprop_step(state_below_in, state_before, W_recurrent, W_in, b):
            pre_h = (state_below_in +
                     tensor.dot(state_before, W_recurrent) + b)
            h = tensor.tanh(pre_h)
            return h

        h, updates = scan(fn=_fprop_step, sequences=[state_below_in],
                          outputs_info=[h0], non_sequences=self.params)
        self._scan_updates.update(updates)

        assert h.ndim == 3
        rval = h[-1]
        assert rval.ndim == 2

        return rval

    def get_params(self):
        return self.params 


class Gated_Recurrent(Recurrent):
    
    def __init__(self, dim, layer_name, irange):
        super(Gated_Recurrent, self).__init__(dim, layer_name, irange)
        self.__dict__.update(locals())
        del self.self

    def set_input_space(self, space):
        super(Gated_Recurrent, self).set_input_space(space)

        # Following the notation in
        # "Learning Phrase Representations using RNN Encoder-Decoder
        # for Statistical Machine Translation", W weighs the input
        # and U weighs the recurrent value.
        W_z = self.rng.uniform(-self.irange, self.irange,
                                  (space.dim, self.dim))
        W_r = self.rng.uniform(-self.irange, self.irange,
                                  (space.dim, self.dim))
        U_z = self.rng.uniform(-self.irange, self.irange,
                                  (self.dim, self.dim))
        U_r = self.rng.uniform(-self.irange, self.irange,
                                  (self.dim, self.dim))
        W_z, W_r = sharedX(W_z), sharedX(W_r)
        U_z, U_r = sharedX(U_z), sharedX(U_r)
        W_z.name, W_r.name, U_z.name, U_r.name = [
            self.layer_name + '_' + param for param in 
            ['W_z', 'W_r', 'U_z', 'U_r']
        ]
        b_z = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_z')
        b_r = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_r')

        self.params.extend([W_z, U_z, b_z, W_r, U_r, b_r])

    def fprop(self, state_below):
        # The initial hidden state is just batch_size zeros 
        print "state_below ndim", state_below.ndim
        h0 = tensor.alloc(np.cast[config.floatX](0),
                          state_below.shape[1], self.dim)
        h0.name = "h0"
        W_in = self.params[1]
        W_z = self.params[3]
        W_r = self.params[6]
        state_below_in = tensor.dot(state_below, W_in)
        state_below_z = tensor.dot(state_below, W_z)
        state_below_r = tensor.dot(state_below, W_r)

        # state_below is the new input, state_before is hidden state
        def _fprop_step(state_below_in, state_below_z, state_below_r, 
                        state_before, W_recurrent, W_in, b,
                        W_z, U_z, b_z, W_r, U_r, b_r):
    
            z = tensor.nnet.sigmoid(state_below_z + tensor.dot(state_before, U_z) + b_z)
            r = tensor.nnet.sigmoid(state_below_r + tensor.dot(state_before, U_r) + b_r)
            pre_h = (
                state_below_in + r * tensor.dot(state_before, W_recurrent)
                + b
            )
            new_h = tensor.tanh(pre_h)
            
            h = z * state_before + (1. - z) * new_h

            return h

        h, updates = scan(fn=_fprop_step, sequences=[state_below_in, state_below_z, state_below_r],
                          outputs_info=[h0], non_sequences=self.params)
        self._scan_updates.update(updates)

        assert h.ndim == 3
        rval = h[-1] #.reshape(state_below.shape[1], self.dim)
        assert rval.ndim == 2

        return rval

class Multiplicative_Gated_Recurrent(mlp.Layer):
    def __init__(self, proj_dim, dim, layer_name, max_labels, irange):
        self.max_labels = max_labels
        self.proj_dim = proj_dim
        self._rnn_friendly = True
        self.__dict__.update(locals())
        del self.self
        super(Multiplicative_Gated_Recurrent, self).__init__()

    def set_input_space(self, space):
        # This space expects an IndexSpace sequence
        assert isinstance(space, SequenceSpace)
        assert isinstance(space.space, IndexSpace)

        # Construct the weight matrices and biases
        self.rng = self.mlp.rng
        W_proj = sharedX(self.rng.uniform(-self.irange, self.irange,
                                       (self.max_labels, self.proj_dim)))
        W_proj.name = self.layer_name + '_proj'
        self.projection = W_proj
      
        W_recurrent = self.rng.uniform(-self.irange, self.irange,
                                       (self.max_labels, self.dim, self.dim)) 
        W_recurrent = np.asarray([scipy.linalg.svd(W)[0] for W in W_recurrent])
        
        W_in = self.rng.uniform(-self.irange, self.irange,
                                (self.proj_dim, self.dim))
        

        W_recurrent, W_in = sharedX(0.9 * W_recurrent), sharedX(W_in)
        W_recurrent.name, W_in.name = [self.layer_name + '_' + param for param in ['W_recurrent', 'W_in']]
        b = sharedX(np.zeros((self.max_labels, self.dim)), name=self.layer_name + '_b') 

        # Save the parameters and set the output space
        self.params = [W_recurrent, W_in, b]
        self.output_space = VectorSpace(dim=self.dim)
        self.input_space = space

        # Following the notation in
        # "Learning Phrase Representations using RNN Encoder-Decoder
        # for Statistical Machine Translation", W weighs the input
        # and U weighs the recurrent value.
        W_z = self.rng.uniform(-self.irange, self.irange,
                                  (self.proj_dim, self.dim))
        W_r = self.rng.uniform(-self.irange, self.irange,
                                  (self.proj_dim, self.dim))
        U_z = self.rng.uniform(-self.irange, self.irange,
                                  (self.dim, self.dim))
        U_r = self.rng.uniform(-self.irange, self.irange,
                                  (self.dim, self.dim))
        W_z, W_r = sharedX(W_z), sharedX(W_r)
        U_z, U_r = sharedX(U_z), sharedX(U_r)
        W_z.name, W_r.name, U_z.name, U_r.name = [
            self.layer_name + '_' + param for param in 
            ['W_z', 'W_r', 'U_z', 'U_r']
        ]
        b_z = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_z')
        b_r = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_r')

        self.params.extend([W_z, U_z, b_z, W_r, U_r, b_r])

    def fprop(self, state_below):
        # The initial hidden state is just batch_size zeros 
        # !!! Move to efficient indexing
        # proj = self.projection.project(state_below)
        #proj = self.project1(self.projection, state_below)

        print "__________________________________"
        shape = state_below.shape
        state_below = state_below.reshape((state_below.shape[0]*state_below.shape[2], state_below.shape[1]))
        proj = self.projection[state_below]
        #print "proj dim", proj.ndim
        
        h0 = tensor.alloc(np.cast[config.floatX](0),
                          shape[1],# 1, 
                          self.dim)
        h0.name = "h0"
        W_in = self.params[1]
        W_z = self.params[3]
        W_r = self.params[6]
        state_below_in = tensor.dot(proj, W_in)
        state_below_z = tensor.dot(proj, W_z)
        state_below_r = tensor.dot(proj, W_r)
        print "h0", h0, h0.dtype, h0.type, h0.ndim, h0.broadcastable
       # print "h0[0]", h0[0], h0[0].dtype, h0[0].type, h0[0].ndim, h0[0].broadcastable
        h0 = tensor.unbroadcast(h0, 0)
        # state_below is the new input, state_before is hidden state
        def _fprop_step(state_below, state_below_in, state_below_z, state_below_r, 
                        state_before, W_recurrent, W_in, b,
                        W_z, U_z, b_z, W_r, U_r, b_r):
            print "state before 1", state_before, state_before.dtype, state_before.type, state_before.broadcastable
            #state_before = tensor.unbroadcast(state_before, 0)
            z = tensor.nnet.sigmoid(state_below_z + tensor.dot(state_before, U_z) + b_z)
            r = tensor.nnet.sigmoid(state_below_r + tensor.dot(state_before, U_r) + b_r)
            #print "r dim", r.type.ndim
            #W_rec = self.project1(W_recurrent, state_below)
            print "State below step", state_below, state_below.broadcastable, state_below.ndim

            print "state before 2", state_before, state_before.dtype, state_before.type, state_before.broadcastable
            W_rec = W_recurrent[state_below]
            
            bias = b[state_below]
            # !!! Move to efficient indexing
            #shape = (state_below.shape[0], state_below.shape[1], self.dim)
            pre_h = (
                state_below_in + r * tensor.batched_dot(state_before, W_rec)#.reshape(shape)
                + bias
            )
            print "pre_h dim", pre_h, pre_h.type.ndim
            #print "W_recurrent[state_below] dim", W_rec, W_rec.ndim
            # print "W_rec * state before", (state_before* W_rec).ndim

            new_h = tensor.tanh(pre_h)
            #print "new_h", new_h
            h = z * state_before + (1. - z) * new_h
            print "final h dim", h, h.type, h.broadcastable, h.ndim
            h = tensor.unbroadcast(h, 0)
            return h

        h, updates = scan(fn=_fprop_step, sequences=[state_below, state_below_in, state_below_z, state_below_r],
                          outputs_info=[h0], non_sequences=self.params)
        print "h", h, h.type, h.broadcastable, h.ndim
        #h = tensor.unbroadcast(h, 1)
        h.name = "h2"
        self._scan_updates.update(updates)

        assert h.ndim == 3
        rval = h[-1] 
        print "rval", rval, rval.type, rval.broadcastable, rval.ndim
        assert rval.ndim == 2

        return rval

    def get_params(self):
        return self.params + [self.projection]
