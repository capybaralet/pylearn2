"""
Recurrent Neural Network Layer
"""

from functools import wraps
import numpy as np
from theano import tensor
from pylearn2.models.mlp import Layer
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.utils import sharedX
from theano import config, scan
from theano.compat.python2x import OrderedDict


class Recurrent(Layer):
    """
    A recurrent neural network layer using the hyperbolic tangent
    activation function, passing on all hidden states or a selection
    of them to the next layer.

    The hidden state is initialized to zeros.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    irange : float
        The input-to-hidden weight matrix is initialized with weights in
        the uniform interval (-irange, irange). The hidden-to-hidden
        matrix weights are sampled in the same manner, unless the argument
        svd is set to True (see below).
    indices : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
        Note: For now only [-1] is supported.
    init_bias : float, optional
        Set an initial bias to be added at each time step. Defaults to 0.
    svd : bool, optional
        Use singular value decomposition to factorize the hidden-to-hidden
        transition matrix with weights in U(-irange, irange) into matrices
        U*s*V, where U is orthogonal. This orthogonal matrix is used to
        initialize the weight matrix. Defaults to True.
    nonlinearity : theano function, optional
        Defaults to tensor.tanh, the non-linearity to be applied to the
        hidden state after each update
    """
    def __init__(self, dim, layer_name, irange, indices=None,
                 init_bias=0., svd=True, nonlinearity=tensor.tanh):
        self.rnn_friendly = True
        self._scan_updates = OrderedDict()
        self.__dict__.update(locals())
        del self.self
        super(Recurrent, self).__init__()

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if (not isinstance(space, SequenceSpace) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Recurrent layer needs a SequenceSpace("
                             "VectorSpace) as input but received  %s instead"
                             % (space))
        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim)
        else:
            self.output_space = SequenceSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        # U is the hidden-to-hidden transition matrix
        U = rng.uniform(-self.irange, self.irange, (self.dim, self.dim))
        if self.svd:
            U = self.mlp.rng.randn(self.dim, self.dim)
            U, s, V = np.linalg.svd(U, full_matrices=True, compute_uv=True)

        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange,
                        (self.input_space.dim, self.dim))

        if hasattr(self.nonlinearity, 'setup_rng'):
            self.nonlinearity.rng = self.mlp.rng
        if hasattr(self.nonlinearity, 'set_input_space'):
            self.nonlinearity.set_input_space(VectorSpace(self.dim))

        self._params = [sharedX(W, name=(self.layer_name + '_W')),
                        sharedX(U, name=(self.layer_name + '_U')),
                        sharedX(np.zeros(self.dim) + self.init_bias,
                                name=self.layer_name + '_b')]

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None, state=None,
                                      targets=None):
        params = self._params
        W, U, b = params[:3]
        sq_W = tensor.sqr(W)
        sq_U = tensor.sqr(U)
        row_norms = tensor.sqrt(sq_W.sum(axis=1))
        col_norms = tensor.sqrt(sq_W.sum(axis=0))
        u_row_norms = tensor.sqrt(sq_U.sum(axis=1))
        u_col_norms = tensor.sqrt(sq_U.sum(axis=0))

        rval = OrderedDict([('W_row_norms_min',  row_norms.min()),
                            ('W_row_norms_mean', row_norms.mean()),
                            ('W_row_norms_max',  row_norms.max()),
                            ('W_col_norms_min',  col_norms.min()),
                            ('W_col_norms_mean', col_norms.mean()),
                            ('W_col_norms_max',  col_norms.max()),
                            ('U_row_norms_min', u_row_norms.min()),
                            ('U_row_norms_mean', u_row_norms.mean()),
                            ('U_row_norms_max', u_row_norms.max()),
                            ('U_col_norms_min', u_col_norms.min()),
                            ('U_col_norms_mean', u_col_norms.mean()),
                            ('U_col_norms_max', u_col_norms.max())])

        if (state is not None) or (state_below is not None):
            if state is None:
                state = self.fprop(state_below)

            mx = state.max(axis=0)
            mean = state.mean(axis=0)
            mn = state.min(axis=0)
            rg = mx - mn

            rval['range_x_max_u'] = rg.max()
            rval['range_x_mean_u'] = rg.mean()
            rval['range_x_min_u'] = rg.min()

            rval['max_x_max_u'] = mx.max()
            rval['max_x_mean_u'] = mx.mean()
            rval['max_x_min_u'] = mx.min()

            rval['mean_x_max_u'] = mean.max()
            rval['mean_x_mean_u'] = mean.mean()
            rval['mean_x_min_u'] = mean.min()

            rval['min_x_max_u'] = mn.max()
            rval['min_x_mean_u'] = mn.mean()
            rval['min_x_min_u'] = mn.min()

        return rval

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        # When random variables are used in the scan function the updates
        # dictionary returned by scan might not be empty, and needs to be
        # added to the updates dictionary before compiling the training
        # function
        if any(key in updates for key in self._scan_updates):
            # Don't think this is possible, but let's check anyway
            raise ValueError("A single shared variable is being updated by "
                             "multiple scan functions")
        updates.update(self._scan_updates)

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        state_below, mask = state_below

        # z0 is the initial hidden state which is (batch size, output dim)
        z0 = tensor.alloc(np.cast[config.floatX](0), state_below.shape[1],
                          self.dim)
        if self.dim == 1:
            # This should fix the bug described in Theano issue #1772
            z0 = tensor.unbroadcast(z0, 1)

        # Later we will add a noise function
        W, U, b = self._params

        # It is faster to do the input-to-hidden matrix multiplications
        # outside of scan
        state_below = tensor.dot(state_below, W) + b

        if hasattr(self.nonlinearity, 'fprop'):
            nonlinearity = self.nonlinearity.fprop
        else:
            nonlinearity = self.nonlinearity

        def fprop_step(state_below, mask, state_before, U):
            z = nonlinearity(state_below +
                             tensor.dot(state_before, U))

            # Only update the state for non-masked data, otherwise
            # just carry on the previous state until the end
            z = mask[:, None] * z + (1 - mask[:, None]) * state_before
            return z

        z, updates = scan(fn=fprop_step, sequences=[state_below, mask],
                          outputs_info=[z0], non_sequences=[U])
        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i] for i in self.indices]
            else:
                return z[self.indices[0]]
        else:
            return (z, mask)

    @wraps(Layer.get_params)
    def get_params(self):
        if hasattr(self.nonlinearity, 'get_params'):
            return self._params + self.nonlinearity.get_params()
        else:
            return self._params

class LSTM(Recurrent):
    """
    Implementation of Long Short-Term Memory proposed by
    S. Hochreiter and J. Schmidhuber in their paper
    "Long short-term memory", Neural Computation, 1997.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    output : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
    irange : float
    init_bias : float
    svd : bool,
    forget_gate_init_bias : float
        Bias for forget gate. Set this variable into high value to force
        the model to learn long-term dependencies.
    input_gate_init_bias : float
    output_gate_init_bias : float
    """
    def __init__(self,
                 forget_gate_init_bias=0.,
                 input_gate_init_bias=0.,
                 output_gate_init_bias=0.,
                 **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.rnn_friendly = True
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        super(LSTM, self).set_input_space(space)

        assert self.irange is not None
        # Output gate switch
        W_x = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.input_space.dim, self.dim))
        W_h = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.dim, self.dim))
        self.O_b = sharedX(np.zeros((self.dim,)) + self.output_gate_init_bias,
                           name=(self.layer_name + '_O_b'))
        self.O_x = sharedX(W_x, name=(self.layer_name + '_O_x'))
        self.O_h = sharedX(W_h, name=(self.layer_name + '_O_h'))
        self.O_c = sharedX(W_h.copy(), name=(self.layer_name + '_O_c'))
        # Input gate switch
        self.I_b = sharedX(np.zeros((self.dim,)) + self.input_gate_init_bias,
                           name=(self.layer_name + '_I_b'))
        self.I_x = sharedX(W_x.copy(), name=(self.layer_name + '_I_x'))
        self.I_h = sharedX(W_h.copy(), name=(self.layer_name + '_I_h'))
        self.I_c = sharedX(W_h.copy(), name=(self.layer_name + '_I_c'))
        # Forget gate switch
        self.F_b = sharedX(np.zeros((self.dim,)) + self.forget_gate_init_bias,
                           name=(self.layer_name + '_F_b'))
        self.F_x = sharedX(W_x.copy(), name=(self.layer_name + '_F_x'))
        self.F_h = sharedX(W_h.copy(), name=(self.layer_name + '_F_h'))
        self.F_c = sharedX(W_h.copy(), name=(self.layer_name + '_F_c'))

    @wraps(Layer.get_params)
    def get_params(self):
        rval = super(LSTM, self).get_params()
        rval += [self.O_b, self.O_x, self.O_h, self.O_c]
        rval += [self.I_b, self.I_x, self.I_h, self.I_c]
        rval += [self.F_b, self.F_x, self.F_h, self.F_c]

        return rval

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        # Is this needed?
        if any(key in updates for key in self._scan_updates):
            # Is this possible? What to do in this case?
            raise ValueError("A single shared variable is being updated by "
                             "multiple scan functions")
        updates.update(self._scan_updates)

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        state_below, mask = state_below

        z0 = tensor.alloc(np.cast[config.floatX](0),
                          state_below.shape[1],
                          self.dim)
        c0 = tensor.alloc(np.cast[config.floatX](0),
                          state_below.shape[1],
                          self.dim)

        if state_below.shape[1] == 1:
            z0 = tensor.unbroadcast(z0, 0)
            c0 = tensor.unbroadcast(c0, 0)

        # Later we will add_noise function
        # Meanwhile leave this part in this way
        W = self.W
        U = self.U
        b = self.b
        state_below_input = tensor.dot(state_below, self.I_x) + self.I_b
        state_below_forget = tensor.dot(state_below, self.F_x) + self.F_b
        state_below_output = tensor.dot(state_below, self.O_x) + self.O_b
        state_below = tensor.dot(state_below, W) + b

        def fprop_step(state_below, state_before, cell_before, U):
            i_on = tensor.nnet.sigmoid(
                state_below_input +
                tensor.dot(state_before, self.I_h) +
                tensor.dot(cell_before, self.I_c)
            )
            f_on = tensor.nnet.sigmoid(
                state_below_forget +
                tensor.dot(state_before, self.F_h) +
                tensor.dot(cell_before, self.F_c)
            )

            c_t = state_below + tensor.dot(state_before, U)
            c_t = f_on * cell_before + i_on * tensor.tanh(c_t)

            o_on = tensor.nnet.sigmoid(
                state_below_output +
                tensor.dot(state_before, self.O_h) +
                tensor.dot(c_t, self.O_c)
            )
            z = o_on * tensor.tanh(c_t)

            return z, c_t

        ((z, c), updates) = scan(fn=fprop_step,
                                 sequences=[state_below,
                                            state_below_input,
                                            state_below_forget,
                                            state_below_output],
                                 outputs_info=[z0, c0],
                                 non_sequences=[U])
        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i] for i in self.indices]
            else:
                return z[self.indices[0]]
        else:
            return z


class ClockworkRecurrent(Recurrent):
    """
    Implementation of Clockwork RNN proposed by
    J. Koutnik, K. Greff, F. Gomez and J. Schmidhuber in their paper
    "A Clockwork RNN", ICML, 2014.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    output : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
    irange : float
    init_bias : float
    svd : bool,
    num_modules :
        Number of modules
    """
    def __init__(self,
                 num_modules=1,
                 **kwargs):
        super(ClockworkRecurrent, self).__init__(**kwargs)
        self.rnn_friendly = True
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        assert isinstance(space, SequenceSpace)
        assert isinstance(space.space, VectorSpace)

        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                self.output_space = CompositeSpace([VectorSpace(dim=self.dim)
                                                    for _ in
                                                    range(len(self.indices))])
            else:
                self.output_space = VectorSpace(dim=self.dim)
        else:
            self.output_space = SequenceSpace(VectorSpace(dim=self.dim))

        rng = self.mlp.rng
        assert self.irange is not None
        if self.num_modules == 1:
            # identical to Recurrent Layer
            U = self.mlp.rng.uniform(-self.irange,
                                     self.irange,
                                     (self.dim, self.dim))
            if self.svd:
                U = self.mlp.rng.randn(self.dim, self.dim)
                U, s, V = np.linalg.svd(U, full_matrices=True, compute_uv=True)

            W = rng.uniform(-self.irange, self.irange,
                            (self.input_space.dim, self.dim))

        else:
            # Use exponentially scaled period
            if isinstance(self.dim, list):
                # So far size of each module should be same

                raise NotImplementedError()
            else:
                # It's restricted to use same dimension for each module.
                # This should be generalized.
                # We will use transposed order which is different from
                # the original paper but will give same result.
                assert self.dim % self.num_modules == 0
                self.module_dim = self.dim / self.num_modules
                if self.irange is not None:
                    W = rng.uniform(-self.irange, self.irange,
                                    (self.input_space.dim, self.dim))

                U = np.zeros((self.dim, self.dim), dtype=config.floatX)
                for i in xrange(self.num_modules):
                    for j in xrange(self.num_modules):
                        if i >= j:
                            u = rng.uniform(-self.irange, self.irange,
                                            (self.module_dim, self.module_dim))
                            if self.svd:
                                u, s, v = np.linalg.svd(u, full_matrices=True,
                                                        compute_uv=True)
                            U[i*self.module_dim:(i+1)*self.module_dim,
                              j*self.module_dim:(j+1)*self.module_dim] = u

        self.W = sharedX(W, name=(self.layer_name + '_W'))
        self.U = sharedX(U, name=(self.layer_name + '_U'))
        self.b = sharedX(np.zeros((self.dim,)) + self.init_bias,
                         name=self.layer_name + '_b')
        nonzero_idx = np.nonzero(U)
        mask_weights = np.zeros(shape=(U.shape), dtype=config.floatX)
        mask_weights[nonzero_idx[0], nonzero_idx[1]] = 1.
        self.mask = sharedX(mask_weights)
        # We consider using power of 2 for exponential scale period
        # However, one can easily set clock-rates of integer k by defining a
        # clock-rate matrix M = k**np.arange(self.num_modules)
        M = 2**np.arange(self.num_modules)
        self.M = sharedX(M, name=(self.layer_name + '_M'))

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):

        if self.U in updates:

            updates[self.U] = updates[self.U] * self.mask

        # Is this needed?
        if any(key in updates for key in self._scan_updates):
            # Is this possible? What to do in this case?
            raise ValueError("A single shared variable is being updated by "
                             "multiple scan functions")
        updates.update(self._scan_updates)

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        state_below, mask = state_below

        z0 = tensor.alloc(np.cast[config.floatX](0),
                          state_below.shape[1],
                          self.dim)

        if state_below.shape[1] == 1:
            z0 = tensor.unbroadcast(z0, 0)

        # Later we will add_noise function
        # Meanwhile leave this part in this way
        W = self.W
        U = self.U
        b = self.b

        idx = tensor.arange(state_below.shape[0])

        def fprop_step(state_below, index, state_before, W, U, b):

            state_now = state_before.copy()
            index = self.num_modules -\
                tensor.nonzero(tensor.mod(index+1, self.M))[0].shape[0]
            this_range = index * self.module_dim
            z = tensor.dot(state_below, W[:, :this_range]) +\
                tensor.dot(state_before, U[:, :this_range]) +\
                b[:this_range]
            z = tensor.tanh(z)
            state_now = tensor.set_subtensor(state_now[:, :this_range], z)

            return state_now

        (z, updates) = scan(fn=fprop_step,
                            sequences=[state_below, idx],
                            outputs_info=[z0],
                            non_sequences=[W, U, b])
        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i] for i in self.indices]
            else:
                return z[self.indices[0]]
        else:
            return z

class RUGatedRecurrent(Recurrent):
    """
    A recurrent neural network layer using the hyperbolic tangent
    activation function, passing on all hidden states or a selection
    of them to the next layer.

    The hidden state is initialized to zeros.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    irange : float
        The input-to-hidden weight matrix is initialized with weights in
        the uniform interval (-irange, irange). The hidden-to-hidden
        matrix weights are sampled in the same manner, unless the argument
        svd is set to True (see below).
    indices : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
        Note: For now only [-1] is supported.
    init_bias : float, optional
        Set an initial bias to be added at each time step. Defaults to 0.
    svd : bool, optional
        Use singular value decomposition to factorize the hidden-to-hidden
        transition matrix with weights in U(-irange, irange) into matrices
        U*s*V, where U is orthogonal. This orthogonal matrix is used to
        initialize the weight matrix. Defaults to True.
    nonlinearity : theano function, optional
        Defaults to tensor.tanh, the non-linearity to be applied to the
        hidden state after each update
    """
    def __init__(self,
                 proj_dim,
                 max_labels,
                 reset_gate_init_bias=0.,
                 update_gate_init_bias=0.,
                 **kwargs):
        super(RUGatedRecurrent, self).__init__(**kwargs)
        self.rnn_friendly = True
        self._proj_dim = proj_dim
        self._max_labels = max_labels
        
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if (not isinstance(space, SequenceSpace) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Gated Recurrent layer needs a SequenceSpace("
                             "VectorSpace) as input but received  %s instead"
                             % (space))
        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim)
        else:
            self.output_space = SequenceSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        # U is the hidden-to-hidden transition tensor 
        # (1 matrix per possible input index)
        if self.svd:
            U = self.mlp.rng.randn(self.dim, self.dim)
            
            U = np.linalg.svd(U, full_matrices=True, compute_uv=True)[0]
        else:
            U = rng.uniform(-self.irange, self.irange, 
                            (self.dim, self.dim))


        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange,
                        (self.input_space.dim, self.dim))

        # Following the notation in
        # "Learning Phrase Representations using RNN Encoder-Decoder
        # for Statistical Machine Translation", W weighs the input
        # and U weighs the recurrent value.
        # z is the update gate, and r is the reset gate.
        W_z = rng.uniform(-self.irange, self.irange,
                               (self.input_space.dim, self.dim))
        W_r = rng.uniform(-self.irange, self.irange,
                               (self.input_space.dim, self.dim))
        U_z = rng.uniform(-self.irange, self.irange,
                               (self.dim, self.dim))
        U_r = rng.uniform(-self.irange, self.irange,
                               (self.dim, self.dim))
    
        b_z = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_z')
        b_r = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_r')

        self._parameters = {'W': sharedX(W, name=(self.layer_name + '_W')),
                        'U': sharedX(U, name=(self.layer_name + '_U')),
                        'b': sharedX(np.zeros(self.dim) + self.init_bias,
                                     name=self.layer_name + '_b'),
                        'Wz': sharedX(W_z, name=(self.layer_name + '_Wz')),
                        'Uz': sharedX(U_z, name=(self.layer_name + '_Uz')),
                        'bz': b_z,
                        'Wr': sharedX(W_r, name=(self.layer_name + '_Wr')),
                        'Ur': sharedX(U_r, name=(self.layer_name + '_Ur')),
                        'br': b_r}

        if hasattr(self.nonlinearity, 'setup_rng'):
            self.nonlinearity.rng = self.mlp.rng
        if hasattr(self.nonlinearity, 'set_input_space'):
            self.nonlinearity.set_input_space(VectorSpace(self.dim))

        # for get_layer_monitoring channels
        self._params = [self._parameters[key] for key in ['W', 'U', 'b']]
        self._all_params = self._parameters.values()

    @wraps(Layer.get_params)
    def get_params(self):
        if hasattr(self.nonlinearity, 'get_params'):
            return self._all_params + self.nonlinearity.get_params()
        else:
            return self._all_params

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        state_below, mask = state_below
        shape = state_below.shape
        #state_below = state_below.reshape((shape[0]*shape[2], shape[1]))
        proj = state_below


        # h0 is the initial hidden state which is (batch size, output dim)
        h0 = tensor.alloc(np.cast[config.floatX](0), shape[1], self.dim)

        if self.dim == 1 or h0.broadcastable[1] == True:
            # This should fix the bug described in Theano issue #1772
            h0 = tensor.unbroadcast(h0, 1)

        # It is faster to do the input-to-hidden matrix multiplications
        # outside of scan
        state_in = (tensor.dot(proj, self._parameters['W']))

        state_z = (tensor.dot(proj, self._parameters['Wz']) 
                   + self._parameters['bz'])
        state_r = (tensor.dot(proj, self._parameters['Wr'])
                   + self._parameters['br'])

        if hasattr(self.nonlinearity, 'fprop'):
            nonlinearity = self.nonlinearity.fprop
        else:
            nonlinearity = self.nonlinearity

        def fprop_step(state_below, mask, state_in, state_z, state_r, 
                       state_before, U, b, Uz, Ur):
            z = tensor.nnet.sigmoid(state_z + tensor.dot(state_before, Uz))
            r = tensor.nnet.sigmoid(state_r + tensor.dot(state_before, Ur))
            
            # The subset of recurrent weight matrices to use this batch
            U_i = U
            b_i = b

            pre_h = nonlinearity(state_in 
                                 + r * tensor.dot(state_before, U_i)
                                 + b_i
            )
            h = z * state_before + (1. - z) * pre_h
            # Only update the state for non-masked data, otherwise
            # just carry on the previous state until the end
            h = mask[:, None] * h + (1 - mask[:, None]) * state_before
            return h

        h, updates = scan(fn=fprop_step, 
                          sequences=[state_below, 
                                     mask, state_in, 
                                     state_z, 
                                     state_r],
                          outputs_info=[h0], 
                          non_sequences=[self._parameters['U'],
                                         self._parameters['b'],
                                         self._parameters['Uz'],
                                         self._parameters['Ur']]
        )

        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [h[i] for i in self.indices]
            else:
                return h[self.indices[0]]
        else:
            return (h, mask)


class MultiplicativeRUGatedRecurrent(Recurrent):
    """
    A recurrent neural network layer using the hyperbolic tangent
    activation function, passing on all hidden states or a selection
    of them to the next layer.

    The hidden state is initialized to zeros.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    irange : float
        The input-to-hidden weight matrix is initialized with weights in
        the uniform interval (-irange, irange). The hidden-to-hidden
        matrix weights are sampled in the same manner, unless the argument
        svd is set to True (see below).
    indices : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
        Note: For now only [-1] is supported.
    init_bias : float, optional
        Set an initial bias to be added at each time step. Defaults to 0.
    svd : bool, optional
        Use singular value decomposition to factorize the hidden-to-hidden
        transition matrix with weights in U(-irange, irange) into matrices
        U*s*V, where U is orthogonal. This orthogonal matrix is used to
        initialize the weight matrix. Defaults to True.
    nonlinearity : theano function, optional
        Defaults to tensor.tanh, the non-linearity to be applied to the
        hidden state after each update
    """
    def __init__(self,
                 proj_dim,
                 max_labels,
                 reset_gate_init_bias=0.,
                 update_gate_init_bias=0.,
                 **kwargs):
        super(MultiplicativeRUGatedRecurrent, self).__init__(**kwargs)
        self.rnn_friendly = True
        self._proj_dim = proj_dim
        self._max_labels = max_labels
        
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if (not isinstance(space, SequenceSpace) or
                not isinstance(space.space, IndexSpace)):
            raise ValueError("Multiplicative Recurrent layer needs a SequenceSpace("
                             "IndexSpace) as input but received  %s instead"
                             % (space))
        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim)
        else:
            self.output_space = SequenceSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        # P is the projection matrix to convert indices to vectors
        P = rng.uniform(-self.irange, self.irange, 
                        (self.max_labels, self.proj_dim))

        # U is the hidden-to-hidden transition tensor 
        # (1 matrix per possible input index)
        if self.svd:
            U = self.mlp.rng.randn(self.max_labels, self.dim, self.dim)
            
            U = np.asarray([np.linalg.svd(
                u, full_matrices=True, compute_uv=True)[0] for u in U])
        else:
            U = rng.uniform(-self.irange, self.irange, 
                            (self.max_labels, self.dim, self.dim))


        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange,
                        (self.proj_dim, self.dim))

        # Following the notation in
        # "Learning Phrase Representations using RNN Encoder-Decoder
        # for Statistical Machine Translation", W weighs the input
        # and U weighs the recurrent value.
        # z is the update gate, and r is the reset gate.
        W_z = rng.uniform(-self.irange, self.irange,
                               (self.proj_dim, self.dim))
        W_r = rng.uniform(-self.irange, self.irange,
                               (self.proj_dim, self.dim))
        U_z = rng.uniform(-self.irange, self.irange,
                               (self.dim, self.dim))
        U_r = rng.uniform(-self.irange, self.irange,
                               (self.dim, self.dim))
    
        b_z = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_z')
        b_r = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_r')

        self._parameters = {'P': sharedX(P, name=self.layer_name + '_P'),
                        'W': sharedX(W, name=(self.layer_name + '_W')),
                        'U': sharedX(U, name=(self.layer_name + '_U')),
                        'b': sharedX(np.zeros((self.max_labels,self.dim)) + self.init_bias,
                                     name=self.layer_name + '_b'),
                        'Wz': sharedX(W_z, name=(self.layer_name + '_Wz')),
                        'Uz': sharedX(U_z, name=(self.layer_name + '_Uz')),
                        'bz': b_z,
                        'Wr': sharedX(W_r, name=(self.layer_name + '_Wr')),
                        'Ur': sharedX(U_r, name=(self.layer_name + '_Ur')),
                        'br': b_r}

        if hasattr(self.nonlinearity, 'setup_rng'):
            self.nonlinearity.rng = self.mlp.rng
        if hasattr(self.nonlinearity, 'set_input_space'):
            self.nonlinearity.set_input_space(VectorSpace(self.dim))

        # for get_layer_monitoring channels
        self._params = [self._parameters[key] for key in ['W', 'U', 'b']]
        self._all_params = self._parameters.values()

    @wraps(Layer.get_params)
    def get_params(self):
        if hasattr(self.nonlinearity, 'get_params'):
            return self._all_params + self.nonlinearity.get_params()
        else:
            return self._all_params

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        state_below, mask = state_below
        shape = state_below.shape
        state_below = state_below.reshape((shape[0]*shape[2], shape[1]))
        proj = self._parameters['P'][state_below]


        # h0 is the initial hidden state which is (batch size, output dim)
        h0 = tensor.alloc(np.cast[config.floatX](0), shape[1], self.dim)

        if self.dim == 1 or h0.broadcastable[1] == True:
            # This should fix the bug described in Theano issue #1772
            h0 = tensor.unbroadcast(h0, 1)

        # It is faster to do the input-to-hidden matrix multiplications
        # outside of scan
        state_in = (tensor.dot(proj, self._parameters['W']))

        state_z = (tensor.dot(proj, self._parameters['Wz']) 
                   + self._parameters['bz'])
        state_r = (tensor.dot(proj, self._parameters['Wr'])
                   + self._parameters['br'])

        if hasattr(self.nonlinearity, 'fprop'):
            nonlinearity = self.nonlinearity.fprop
        else:
            nonlinearity = self.nonlinearity

        def fprop_step(state_below, mask, state_in, state_z, state_r, 
                       state_before, U, b, Uz, Ur):
            z = tensor.nnet.sigmoid(state_z + tensor.dot(state_before, Uz))
            r = tensor.nnet.sigmoid(state_r + tensor.dot(state_before, Ur))
            
            # The subset of recurrent weight matrices to use this batch
            U_i = U[state_below]
            b_i = b[state_below]

            pre_h = nonlinearity(state_in 
                                 + r * tensor.batched_dot(state_before, U_i)
                                 + b_i
            )
            h = z * state_before + (1. - z) * pre_h
            # Only update the state for non-masked data, otherwise
            # just carry on the previous state until the end
            h = mask[:, None] * h + (1 - mask[:, None]) * state_before
            return h

        h, updates = scan(fn=fprop_step, 
                          sequences=[state_below, 
                                     mask, state_in, 
                                     state_z, 
                                     state_r],
                          outputs_info=[h0], 
                          non_sequences=[self._parameters['U'],
                                         self._parameters['b'],
                                         self._parameters['Uz'],
                                         self._parameters['Ur']]
        )

        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [h[i] for i in self.indices]
            else:
                return h[self.indices[0]]
        else:
            return (h, mask)

class FactoredMultiplicativeRUGatedRecurrent(Recurrent):
    """
    A recurrent neural network layer using the hyperbolic tangent
    activation function, passing on all hidden states or a selection
    of them to the next layer.

    The hidden state is initialized to zeros.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    irange : float
        The input-to-hidden weight matrix is initialized with weights in
        the uniform interval (-irange, irange). The hidden-to-hidden
        matrix weights are sampled in the same manner, unless the argument
        svd is set to True (see below).
    indices : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
        Note: For now only [-1] is supported.
    init_bias : float, optional
        Set an initial bias to be added at each time step. Defaults to 0.
    svd : bool, optional
        Use singular value decomposition to factorize the hidden-to-hidden
        transition matrix with weights in U(-irange, irange) into matrices
        U*s*V, where U is orthogonal. This orthogonal matrix is used to
        initialize the weight matrix. Defaults to True.
    nonlinearity : theano function, optional
        Defaults to tensor.tanh, the non-linearity to be applied to the
        hidden state after each update
    """
    def __init__(self,
                 proj_dim,
                 max_labels,
                 reset_gate_init_bias=0.,
                 update_gate_init_bias=0.,
                 **kwargs):
        super(FactoredMultiplicativeRUGatedRecurrent, self).__init__(**kwargs)
        self.rnn_friendly = True
        self._max_labels = max_labels # d
        
        self.__dict__.update(locals())
        del self.self
        print "______________"
        print "RNN COST", self.cost

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if (not isinstance(space, SequenceSpace) or
                not isinstance(space.space, IndexSpace)):
            raise ValueError("Multiplicative Recurrent layer needs a SequenceSpace("
                             "IndexSpace) as input but received  %s instead"
                             % (space))
        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim)
        else:
            self.output_space = SequenceSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        # P is the projection matrix to convert indices to vectors
        P = rng.uniform(-self.irange, self.irange, 
                        (self.max_labels, self.proj_dim))

        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange,
                        (self.proj_dim, self.proj_dim))


        # U is the hidden-to-hidden transition tensor 
        if self.svd:
            U = self.mlp.rng.randn(self.dim, self.proj_dim)
            
            #U, _ , _ = np.linalg.svd(U, full_matrices=True, compute_uv=True)
        else:
            U = rng.uniform(-self.irange, self.irange, 
                            (self.dim, self.proj_dim))

        # V is the hidden-to-hidden transition tensor 
        if self.svd:
            V = self.mlp.rng.randn(self.proj_dim, self.dim)
            
            #V, _, _ = np.linalg.svd(V, full_matrices=True, compute_uv=True)
        else:
            V = rng.uniform(-self.irange, self.irange, 
                            (self.proj_dim, self.dim))

        # # W is the input-to-hidden matrix
        # W = rng.uniform(-self.irange, self.irange,
        #                 (self._proj_dim, self.dim))

        # Following the notation in
        # "Learning Phrase Representations using RNN Encoder-Decoder
        # for Statistical Machine Translation", W weighs the input
        # and U weighs the recurrent value.
        # z is the update gate, and r is the reset gate.
        W_z = rng.uniform(-self.irange, self.irange,
                               (self.proj_dim, self.dim))
        W_r = rng.uniform(-self.irange, self.irange,
                               (self.proj_dim, self.dim))
        U_z = rng.uniform(-self.irange, self.irange,
                               (self.dim, self.dim))
        U_r = rng.uniform(-self.irange, self.irange,
                               (self.dim, self.dim))
    
        b_z = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_z')
        b_r = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_r')

        self._parameters = {'P': sharedX(P, name=self.layer_name + '_P'),
                        'W': sharedX(W, name=(self.layer_name + '_W')),
                        'V': sharedX(V, name=(self.layer_name + '_V')),
                        'U': sharedX(U, name=(self.layer_name + '_U')),
                        'b': sharedX(np.zeros((self.dim,)) + self.init_bias,
                                     name=self.layer_name + '_b'),
                        'Wz': sharedX(W_z, name=(self.layer_name + '_Wz')),
                        'Uz': sharedX(U_z, name=(self.layer_name + '_Uz')),
                        'bz': b_z,
                        'Wr': sharedX(W_r, name=(self.layer_name + '_Wr')),
                        'Ur': sharedX(U_r, name=(self.layer_name + '_Ur')),
                        'br': b_r}

        # for get_layer_monitoring channels
        self._params = [self._parameters[key] for key in ['W', 'U', 'b']]

    @wraps(Layer.get_params)
    def get_params(self):
        return self._parameters.values()
        
    @wraps(Layer.fprop)
    def fprop(self, state_below):
        state_below, mask = state_below
        shape = state_below.shape
        state_below = state_below.reshape((shape[0]*shape[2], shape[1]))
        proj = self._parameters['P'][state_below]
        

        # h0 is the initial hidden state which is (batch size, output dim)
        h0 = tensor.alloc(np.cast[config.floatX](0), shape[1], self.dim)

        if self.dim == 1 or h0.broadcastable[1] == True:
            # This should fix the bug described in Theano issue #1772
            h0 = tensor.unbroadcast(h0, 1)

        # It is faster to do the input-to-hidden matrix multiplications
        # outside of scan
        state_in = (tensor.dot(proj, self._parameters['W']))

        state_z = (tensor.dot(proj, self._parameters['Wz']) 
                         + self._parameters['bz'])
        state_r = (tensor.dot(proj, self._parameters['Wr'])
                         + self._parameters['br'])

        def fprop_step(mask, state_in, state_z, state_r, 
                       state_before, U, V, b, Uz, Ur):
            z = tensor.nnet.sigmoid(state_z + tensor.dot(state_before, Uz))
            r = tensor.nnet.sigmoid(state_r + tensor.dot(state_before, Ur))
            
            # The subset of recurrent weight matrices to use this batch
            pre_h1 = tensor.dot(state_before, U)
            pre_h2 = pre_h1 * state_in
            h = self.nonlinearity(r*tensor.dot(pre_h2, V) +b)
            h = z * state_before + (1. - z) * h
            # Only update the state for non-masked data, otherwise
            # just carry on the previous state until the end
            h = mask[:, None] * h + (1 - mask[:, None]) * state_before
            return h

        h, updates = scan(fn=fprop_step, 
                          sequences=[mask, state_in, 
                                     state_z, 
                                     state_r],
                          outputs_info=[h0], 
                          non_sequences=[self._parameters['U'],
                                         self._parameters['V'],
                                         self._parameters['b'],
                                         self._parameters['Uz'],
                                         self._parameters['Ur']]
        )

        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [h[i] for i in self.indices]
            else:
                return h[self.indices[0]]
        else:
            return (h, mask)

class Words_And_FactoredMultiplicativeRUGatedRecurrent(Recurrent):
    """
    A recurrent neural network layer using the hyperbolic tangent
    activation function, passing on all hidden states or a selection
    of them to the next layer.

    The hidden state is initialized to zeros.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        Initializes each weight randomly in U(-irange, irange)
    irange : float
        The input-to-hidden weight matrix is initialized with weights in
        the uniform interval (-irange, irange). The hidden-to-hidden
        matrix weights are sampled in the same manner, unless the argument
        svd is set to True (see below).
    indices : slice, list of integers or integer, optional
        If specified this layer will return only the given hidden
        states. If an integer is given, it will not return a
        SequenceSpace. Otherwise, it will return a SequenceSpace of
        fixed length. Note that a SequenceSpace of fixed length
        can be flattened by using the FlattenerLayer.
        Note: For now only [-1] is supported.
    init_bias : float, optional
        Set an initial bias to be added at each time step. Defaults to 0.
    svd : bool, optional
        Use singular value decomposition to factorize the hidden-to-hidden
        transition matrix with weights in U(-irange, irange) into matrices
        U*s*V, where U is orthogonal. This orthogonal matrix is used to
        initialize the weight matrix. Defaults to True.
    nonlinearity : theano function, optional
        Defaults to tensor.tanh, the non-linearity to be applied to the
        hidden state after each update
    """
    def __init__(self,
                 proj_dim,
                 char_labels,
                 word_labels,
                 reset_gate_init_bias=0.,
                 update_gate_init_bias=0.,
                 **kwargs):
        super(Words_And_FactoredMultiplicativeRUGatedRecurrent, self).__init__(**kwargs)
        self.rnn_friendly = True
        self._max_labels = char_labels # d
        
        self.__dict__.update(locals())
        del self.self
        
    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if (not isinstance(space, CompositeSpace) or
                not isinstance(space.components[0].space, IndexSpace)):
            raise ValueError("Multiplicative Recurrent layer needs a CompositeSpace(SequenceSpace("
                             "IndexSpace), IndexSpace) as input but received  %s instead"
                             % (space))
        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim*2)
        else:
            self.output_space = SequenceSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        # P is the projection matrix to convert indices to vectors
        P = rng.uniform(-self.irange, self.irange, 
                        (self.char_labels, self.proj_dim))

        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange,
                        (self.proj_dim, self.proj_dim))


        # U is the hidden-to-hidden transition tensor 
        if self.svd:
            U = self.mlp.rng.randn(self.dim, self.proj_dim)
            
            #U, _ , _ = np.linalg.svd(U, full_matrices=True, compute_uv=True)
        else:
            U = rng.uniform(-self.irange, self.irange, 
                            (self.dim, self.proj_dim))

        # V is the hidden-to-hidden transition tensor 
        if self.svd:
            V = self.mlp.rng.randn(self.proj_dim, self.dim)
            
            #V, _, _ = np.linalg.svd(V, full_matrices=True, compute_uv=True)
        else:
            V = rng.uniform(-self.irange, self.irange, 
                            (self.proj_dim, self.dim))

        # # W is the input-to-hidden matrix
        # W = rng.uniform(-self.irange, self.irange,
        #                 (self._proj_dim, self.dim))

        # Following the notation in
        # "Learning Phrase Representations using RNN Encoder-Decoder
        # for Statistical Machine Translation", W weighs the input
        # and U weighs the recurrent value.
        # z is the update gate, and r is the reset gate.
        W_z = rng.uniform(-self.irange, self.irange,
                               (self.proj_dim, self.dim))
        W_r = rng.uniform(-self.irange, self.irange,
                               (self.proj_dim, self.dim))
        U_z = rng.uniform(-self.irange, self.irange,
                               (self.dim, self.dim))
        U_r = rng.uniform(-self.irange, self.irange,
                               (self.dim, self.dim))
    

        b_z = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_z')
        b_r = sharedX(np.zeros((self.dim,)), name=self.layer_name + '_b_r')

        wordProj = rng.uniform(-self.irange, self.irange, (self.word_labels, self.dim))

        self._parameters = {'P': sharedX(P, name=self.layer_name + '_P'),
                            'W': sharedX(W, name=(self.layer_name + '_W')),
                            'V': sharedX(V, name=(self.layer_name + '_V')),
                            'U': sharedX(U, name=(self.layer_name + '_U')),
                            'b': sharedX(np.zeros((self.dim,)) + self.init_bias,
                                         name=self.layer_name + '_b'),
                            'Wz': sharedX(W_z, name=(self.layer_name + '_Wz')),
                            'Uz': sharedX(U_z, name=(self.layer_name + '_Uz')),
                            'bz': b_z,
                            'Wr': sharedX(W_r, name=(self.layer_name + '_Wr')),
                            'Ur': sharedX(U_r, name=(self.layer_name + '_Ur')),
                            'br': b_r,
                            'WP' : sharedX(wordProj, name=(self.layer_name + '_WP')),
        }

        # for get_layer_monitoring channels
        self._params = [self._parameters[key] for key in ['W', 'U', 'b']]

    @wraps(Layer.get_params)
    def get_params(self):
        return self._parameters.values()
        
    @wraps(Layer.fprop)
    def fprop(self, state_below):
        import pdb
        pdb.set_trace()
        state_below, mask, words = state_below
        shape = state_below.shape
        state_below = state_below.reshape((shape[0]*shape[2], shape[1]))
        proj = self._parameters['P'][state_below]
        wordproj = self._parameters['WP'][words]

        # h0 is the initial hidden state which is (batch size, output dim)
        h0 = tensor.alloc(np.cast[config.floatX](0), shape[1], self.dim)

        if self.dim == 1 or h0.broadcastable[1] == True:
            # This should fix the bug described in Theano issue #1772
            h0 = tensor.unbroadcast(h0, 1)

        # It is faster to do the input-to-hidden matrix multiplications
        # outside of scan
        state_in = (tensor.dot(proj, self._parameters['W']))

        state_z = (tensor.dot(proj, self._parameters['Wz']) 
                         + self._parameters['bz'])
        state_r = (tensor.dot(proj, self._parameters['Wr'])
                         + self._parameters['br'])

        def fprop_step(mask, state_in, state_z, state_r, 
                       state_before, U, V, b, Uz, Ur):
            z = tensor.nnet.sigmoid(state_z + tensor.dot(state_before, Uz))
            r = tensor.nnet.sigmoid(state_r + tensor.dot(state_before, Ur))
            
            # The subset of recurrent weight matrices to use this batch
            pre_h1 = tensor.dot(state_before, U)
            pre_h2 = pre_h1 * state_in
            h = self.nonlinearity(r*tensor.dot(pre_h2, V) +b)
            h = z * state_before + (1. - z) * h
            # Only update the state for non-masked data, otherwise
            # just carry on the previous state until the end
            h = mask[:, None] * h + (1 - mask[:, None]) * state_before
            return h

        h, updates = scan(fn=fprop_step, 
                          sequences=[mask, state_in, 
                                     state_z, 
                                     state_r],
                          outputs_info=[h0], 
                          non_sequences=[self._parameters['U'],
                                         self._parameters['V'],
                                         self._parameters['b'],
                                         self._parameters['Uz'],
                                         self._parameters['Ur']]
        )

        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                charproj =  [h[i] for i in self.indices]
            else:
                charproj =  h[self.indices[0]]
            return tensor.concatenate([charproj, wordproj], axis=1)
        else:
            return (tensor.concatenate([h, wordproj], axis=1), mask)

class RecurrentLeakyNeurons(Recurrent):

    def __init__(self,
                 num_modules=2,
                 update_gate_init_bias=0.,
                 reset_gate_init_bias=0.,
                 **kwargs):
        super(RecurrentLeakyNeurons, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    def set_input_space(self, space):

        if (not isinstance(space, SequenceSpace) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Recurrent layer needs a SequenceSpace("
                             "VectorSpace) as input but received  %s instead"
                             % (space))
        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim)
        else:
            self.output_space = SequenceSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        # So far size of each module should be same
        # It's restricted to use same dimension for each module.
        # It should be generalized.
        # We will use transposed order which is different from
        # the original paper but will give same result.
        assert self.dim % self.num_modules == 0
        self.module_dim = self.dim / self.num_modules

        U = np.zeros((self.dim, self.dim), dtype=config.floatX)
        for i in xrange(self.num_modules):
            for j in xrange(self.num_modules):
                if not((j >= i + 2) and (i < self.num_modules - 2)):
                    if False: #self.istdev is not None:
                        pass
                        # u = rng.randn(self.module_dim,
                        #               self.module_dim)
                        # u *= self.istdev
                    else:
                        u = rng.uniform(-self.irange,
                                        self.irange,
                                        (self.module_dim,
                                         self.module_dim))  #  *\
                            # (rng.uniform(0., 1., (self.module_dim,
                            #                       self.module_dim))
                            #  < self.include_prob)
                    if self.svd:
                        u, s, v = np.linalg.svd(u,
                                                full_matrices=True,
                                                compute_uv=True)
                    U[i*self.module_dim:(i+1)*self.module_dim,
                      j*self.module_dim:(j+1)*self.module_dim] = u
        if True: #self.use_bias:
            self.b = sharedX(np.zeros((self.dim,)) + self.init_bias,
                             name=(self.layer_name + '_b'))
        else:
            assert self.b_lr_scale is None
        self.U = sharedX(U, name=(self.layer_name + '_U'))
        nonzero_idx = np.nonzero(U)
        mask_weights = np.zeros(shape=(U.shape), dtype=config.floatX)
        mask_weights[nonzero_idx[0], nonzero_idx[1]] = 1.
        self.mask = sharedX(mask_weights)

        gate_dim = self.input_space.dim + self.module_dim * (self.num_modules - 1)
        self.gdim = []
        self.gdim.append(np.arange(self.input_space.dim))
        start = self.input_space.dim
        for i in xrange(self.num_modules - 1):
            self.gdim.append(np.arange(start, start + self.module_dim))
            start += self.module_dim

        if self.irange is not None:
            #assert self.istdev is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (gate_dim,
                             self.module_dim))#  *\
                # (rng.uniform(0., 1., (gate_dim,
                #                       self.module_dim))
                #  < self.include_prob)
        # elif self.istdev is not None:
        #     W = rng.randn(gate_dim,
        #                   self.module_dim) * self.istdev
        self.W = sharedX(W, name=(self.layer_name + '_W'))
        if self.irange is not None:
            #assert self.istdev is None
            S = rng.uniform(-self.irange,
                            self.irange,
                            (self.num_modules - 1,
                             self.input_space.dim,
                             self.module_dim))#  *\
                # (rng.uniform(0., 1., (self.num_modules - 1,
                #                       self.input_space.dim,
                #                       self.module_dim))
                #  < self.include_prob)
        # elif self.istdev is not None:
        #     S = rng.randn(self.num_modules - 1,
        #                   self.input_space.dim,
        #                   self.module_dim) * self.istdev
        self.S = sharedX(S, name=(self.layer_name + '_S'))

        # Reset gate switch
        R_s = rng.uniform(-self.irange,
                          self.irange,
                          (self.num_modules - 1,
                           self.input_space.dim,
                           self.module_dim))
        R_x = rng.uniform(-self.irange,
                          self.irange,
                          (gate_dim,
                           self.module_dim))
        R_h = rng.uniform(-self.irange,
                          self.irange,
                          (self.num_modules,
                           self.dim,
                           self.module_dim))
        self.R_b = sharedX(np.zeros((self.num_modules,
                                     self.module_dim)) +
                           self.reset_gate_init_bias,
                           name=(self.layer_name + '_R_b'))
        self.R_s = sharedX(R_s, name=(self.layer_name + '_R_s'))
        self.R_x = sharedX(R_x, name=(self.layer_name + '_R_x'))
        self.R_h = sharedX(R_h, name=(self.layer_name + '_R_h'))
        # Update gate switch
        U_s = rng.uniform(-self.irange,
                          self.irange,
                          (self.num_modules - 1,
                           self.input_space.dim,
                           self.module_dim))
        U_x = rng.uniform(-self.irange,
                          self.irange,
                          (self.num_modules - 1,
                           self.module_dim,
                           self.module_dim))
        U_h = rng.uniform(-self.irange,
                          self.irange,
                          (self.num_modules - 1,
                           self.dim,
                           self.module_dim))
        self.U_b = sharedX(np.zeros((self.num_modules - 1,
                                     self.module_dim)) +
                           self.update_gate_init_bias,
                           name=(self.layer_name + '_U_b'))
        self.U_s = sharedX(U_s, name=(self.layer_name + '_U_s'))
        self.U_x = sharedX(U_x, name=(self.layer_name + '_U_x'))
        self.U_h = sharedX(U_h, name=(self.layer_name + '_U_h'))
        self._params = [self.U, self.W, self.b]

    def get_params(self):
        rval = super(RecurrentLeakyNeurons, self).get_params()
        rval += [self.R_s, self.R_x, self.R_h, self.R_b]
        rval += [self.U_s, self.U_x, self.U_h, self.U_b]
        rval += [self.S]

        return rval

    def _modify_updates(self, updates):

        # if self.max_row_norm is not None:
        #     if self.W in updates:
        #         updated_W = updates[self.W]
        #         row_norms = tensor.sqrt(tensor.sum(tensor.sqr(updated_W), axis=1))
        #         desired_norms = tensor.clip(row_norms, 0, self.max_row_norm)
        #         scales = desired_norms / (1e-7 + row_norms)
        #         updates[self.W] = updated_W * scales.dimshuffle(0, 'x')
        # if self.max_col_norm is not None or self.min_col_norm is not None:
        #     assert self.max_row_norm is None
        #     if self.max_col_norm is not None:
        #         max_col_norm = self.max_col_norm
        #     if self.min_col_norm is None:
        #         self.min_col_norm = 0
        #     if self.W in updates:
        #         updated_W = updates[self.W]
        #         col_norms = tensor.sqrt(tensor.sum(tensor.sqr(updated_W), axis=0))
        #         if self.max_col_norm is None:
        #             max_col_norm = col_norms.max()
        #         desired_norms = tensor.clip(col_norms,
        #                                     self.min_col_norm,
        #                                     max_col_norm)
        #         updates[self.W] = updated_W * desired_norms/(1e-7 + col_norms)

        if self.U in updates:
            updates[self.U] = updates[self.U] * self.mask

    def fprop(self, state_below):
        state_below, mask = state_below

        # z0 is the initial hidden state which is (batch size, output dim)
        z0 = tensor.alloc(np.cast[config.floatX](0), state_below.shape[1],
                          self.dim)
        if self.dim == 1:
            # This should fix the bug described in Theano issue #1772
            z0 = tensor.unbroadcast(z0, 1)

        # Later we will add a noise function
        W = self.W
        U = self.U

        def fprop_step(state_below, mask, state_before, W, U):

            r = tensor.dot(state_before, U)
            h, z = self.first_step(state_below, state_before, r, W, 0)
            h, z = self.gated_step(state_below, h, z, r, W, 1)
            h, z = self.gated_step(state_below, h, z, r, W, 2)

            # Only update the state for non-masked data, otherwise
            # just carry on the previous state until the end
            z = mask[:, None] * z + (1 - mask[:, None]) * state_before
            return z

        z, updates = scan(fn=fprop_step,
                          sequences=[state_below, mask],
                          outputs_info=[z0],
                          non_sequences=[W, U])

        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i] for i in self.indices]
            else:
                return z[self.indices[0]]
        else:
            return (z, mask)

    def first_step(self, state_below, state_before, r, W, clk):

        c0 = clk * self.module_dim
        c1 = (clk + 1) * self.module_dim

        r_on = tensor.nnet.sigmoid(tensor.dot(state_below,
                                              self.R_x[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +
                                   tensor.dot(state_before, self.R_h[clk]) +
                                   self.R_b[clk])
        pre_h = r_on * r[:, c0:c1] +\
            tensor.dot(state_below,
                       W[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +\
            self.b[c0:c1]

        h = tensor.tanh(pre_h)
        z = tensor.set_subtensor(state_before[:, c0:c1], h)

        return h, z

    def gated_step(self, input_t, state_below, state_before, r, W, clk):

        cm1 = (clk - 1) * self.module_dim
        c0 = clk * self.module_dim
        c1 = (clk + 1) * self.module_dim

        r_on = tensor.nnet.sigmoid(tensor.dot(input_t, self.R_s[clk - 1]) +
                                   tensor.dot(state_below,
                                   self.R_x[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +
                                   tensor.dot(state_before[:, cm1:], self.R_h[clk][cm1:, :]) +
                                   self.R_b[clk])
        pre_h = r_on * r[:, c0:c1] +\
            tensor.dot(state_below,
                       W[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +\
            self.b[c0:c1] + tensor.dot(input_t, self.S[clk - 1])

        u_on = tensor.nnet.sigmoid(tensor.dot(input_t, self.U_s[clk - 1]) +
                                   tensor.dot(state_below, self.U_x[clk - 1]) +
                                   tensor.dot(state_before[:, cm1:], self.U_h[clk - 1][cm1:, :]) +
                                   self.U_b[clk - 1])

        h = u_on * state_before[:, c0:c1] + (1. - u_on) * tensor.tanh(pre_h)
        z = tensor.set_subtensor(state_before[:, c0:c1], h)

        return h, z

class RecurrentLeakyNeurons_m(Recurrent):

    def __init__(self,
                 num_modules=2,
                 update_gate_init_bias=0.,
                 reset_gate_init_bias=0.,
                 **kwargs):
        super(RecurrentLeakyNeurons_m, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    def set_input_space(self, space):

        if (not isinstance(space, SequenceSpace) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Recurrent layer needs a SequenceSpace("
                             "VectorSpace) as input but received  %s instead"
                             % (space))
        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim)
        else:
            self.output_space = SequenceSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        # So far size of each module should be same
        # It's restricted to use same dimension for each module.
        # It should be generalized.
        # We will use transposed order which is different from
        # the original paper but will give same result.
        assert self.dim % self.num_modules == 0
        self.module_dim = self.dim / self.num_modules

        U = np.zeros((self.dim, self.dim), dtype=config.floatX)
        for i in xrange(self.num_modules):
            for j in xrange(self.num_modules):
                if not((j >= i + 2) and (i < self.num_modules - 2)):
                    if False: #self.istdev is not None:
                        pass
                        # u = rng.randn(self.module_dim,
                        #               self.module_dim)
                        # u *= self.istdev
                    else:
                        u = rng.uniform(-self.irange,
                                        self.irange,
                                        (self.module_dim,
                                         self.module_dim))  #  *\
                            # (rng.uniform(0., 1., (self.module_dim,
                            #                       self.module_dim))
                            #  < self.include_prob)
                    if self.svd:
                        u, s, v = np.linalg.svd(u,
                                                full_matrices=True,
                                                compute_uv=True)
                    U[i*self.module_dim:(i+1)*self.module_dim,
                      j*self.module_dim:(j+1)*self.module_dim] = u
        if True: #self.use_bias:
            self.b = sharedX(np.zeros((self.dim,)) + self.init_bias,
                             name=(self.layer_name + '_b'))
        else:
            assert self.b_lr_scale is None
        self.U = sharedX(U, name=(self.layer_name + '_U'))
        nonzero_idx = np.nonzero(U)
        mask_weights = np.zeros(shape=(U.shape), dtype=config.floatX)
        mask_weights[nonzero_idx[0], nonzero_idx[1]] = 1.
        self.mask = sharedX(mask_weights)

        gate_dim = self.input_space.dim + self.module_dim * (self.num_modules - 1)
        self.gdim = []
        self.gdim.append(np.arange(self.input_space.dim))
        start = self.input_space.dim
        for i in xrange(self.num_modules - 1):
            self.gdim.append(np.arange(start, start + self.module_dim))
            start += self.module_dim

        if self.irange is not None:
            #assert self.istdev is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (gate_dim,
                             self.module_dim))#  *\
                # (rng.uniform(0., 1., (gate_dim,
                #                       self.module_dim))
                #  < self.include_prob)
        # elif self.istdev is not None:
        #     W = rng.randn(gate_dim,
        #                   self.module_dim) * self.istdev
        self.W = sharedX(W, name=(self.layer_name + '_W'))
        if self.irange is not None:
            #assert self.istdev is None
            S = rng.uniform(-self.irange,
                            self.irange,
                            (self.num_modules - 1,
                             self.input_space.dim,
                             self.module_dim))#  *\
                # (rng.uniform(0., 1., (self.num_modules - 1,
                #                       self.input_space.dim,
                #                       self.module_dim))
                #  < self.include_prob)
        # elif self.istdev is not None:
        #     S = rng.randn(self.num_modules - 1,
        #                   self.input_space.dim,
        #                   self.module_dim) * self.istdev
        self.S = sharedX(S, name=(self.layer_name + '_S'))

        # Reset gate switch
        R_s = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.num_modules - 1,
                                    self.input_space.dim,
                                    self.module_dim))
        R_x = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.num_modules - 1,
                                    self.module_dim,
                                    self.module_dim))
        R_h = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.num_modules - 1,
                                    self.dim,
                                    self.module_dim))
        self.R_b = sharedX(np.zeros((self.num_modules - 1,
                                     self.module_dim)) +
                           self.reset_gate_init_bias,
                           name=(self.layer_name + '_R_b'))
        self.R_s = sharedX(R_s, name=(self.layer_name + '_R_s'))
        self.R_x = sharedX(R_x, name=(self.layer_name + '_R_x'))
        self.R_h = sharedX(R_h, name=(self.layer_name + '_R_h'))
        # Update gate switch
        U_s = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.num_modules - 1,
                                    self.input_space.dim,
                                    self.module_dim))
        U_x = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.num_modules - 1,
                                    self.module_dim,
                                    self.module_dim))
        U_h = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.num_modules - 1,
                                    self.dim,
                                    self.module_dim))
        self.U_b = sharedX(np.zeros((self.num_modules - 1,
                                     self.module_dim)) +
                           self.update_gate_init_bias,
                           name=(self.layer_name + '_U_b'))
        self.U_s = sharedX(U_s, name=(self.layer_name + '_U_s'))
        self.U_x = sharedX(U_x, name=(self.layer_name + '_U_x'))
        self.U_h = sharedX(U_h, name=(self.layer_name + '_U_h'))
        self._params = [self.U, self.W, self.b]

    def get_params(self):
        rval = super(RecurrentLeakyNeurons_m, self).get_params()
        rval += [self.R_s, self.R_x, self.R_h, self.R_b]
        rval += [self.U_s, self.U_x, self.U_h, self.U_b]
        rval += [self.S]

        return rval

    def _modify_updates(self, updates):

        # if self.max_row_norm is not None:
        #     if self.W in updates:
        #         updated_W = updates[self.W]
        #         row_norms = tensor.sqrt(tensor.sum(tensor.sqr(updated_W), axis=1))
        #         desired_norms = tensor.clip(row_norms, 0, self.max_row_norm)
        #         scales = desired_norms / (1e-7 + row_norms)
        #         updates[self.W] = updated_W * scales.dimshuffle(0, 'x')
        # if self.max_col_norm is not None or self.min_col_norm is not None:
        #     assert self.max_row_norm is None
        #     if self.max_col_norm is not None:
        #         max_col_norm = self.max_col_norm
        #     if self.min_col_norm is None:
        #         self.min_col_norm = 0
        #     if self.W in updates:
        #         updated_W = updates[self.W]
        #         col_norms = tensor.sqrt(tensor.sum(tensor.sqr(updated_W), axis=0))
        #         if self.max_col_norm is None:
        #             max_col_norm = col_norms.max()
        #         desired_norms = tensor.clip(col_norms,
        #                                     self.min_col_norm,
        #                                     max_col_norm)
        #         updates[self.W] = updated_W * desired_norms/(1e-7 + col_norms)

        if self.U in updates:
            updates[self.U] = updates[self.U] * self.mask

    def fprop(self, state_below):
        state_below, mask = state_below

        # z0 is the initial hidden state which is (batch size, output dim)
        z0 = tensor.alloc(np.cast[config.floatX](0), state_below.shape[1],
                          self.dim)
        if self.dim == 1:
            # This should fix the bug described in Theano issue #1772
            z0 = tensor.unbroadcast(z0, 1)

        # Later we will add a noise function
        W = self.W
        U = self.U

        def fprop_step(state_below, mask, state_before, W, U):

            r = tensor.dot(state_before, U)
            h, z = self.first_step(state_below, state_before, r, W, 0)
            h, z = self.gated_step(state_below, h, z, r, W, 1)
            h, z = self.gated_step(state_below, h, z, r, W, 2)

            # Only update the state for non-masked data, otherwise
            # just carry on the previous state until the end
            z = mask[:, None] * z + (1 - mask[:, None]) * state_before
            return z

        z, updates = scan(fn=fprop_step,
                          sequences=[state_below, mask],
                          outputs_info=[z0],
                          non_sequences=[W, U])

        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i] for i in self.indices]
            else:
                return z[self.indices[0]]
        else:
            return (z, mask)

    def first_step(self, state_below, state_before, r, W, clk):

        c0 = clk * self.module_dim
        c1 = (clk + 1) * self.module_dim
        pre_h = r[:, c0:c1] +\
            tensor.dot(state_below,
                       W[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +\
            self.b[c0:c1]

        h = tensor.tanh(pre_h)
        z = tensor.set_subtensor(state_before[:, c0:c1], h)

        return h, z

    def gated_step(self, input_t, state_below, state_before, r, W, clk):

        cm1 = (clk - 1) * self.module_dim
        c0 = clk * self.module_dim
        c1 = (clk + 1) * self.module_dim

        r_on = tensor.nnet.sigmoid(tensor.dot(input_t, self.R_s[clk - 1]) +
                                   tensor.dot(state_below, self.R_x[clk - 1]) +
                                   tensor.dot(state_before[:, cm1:], self.R_h[clk - 1][cm1:, :]) +
                                   self.R_b[clk - 1])
        pre_h = r_on * r[:, c0:c1] +\
            tensor.dot(state_below,
                       W[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +\
            self.b[c0:c1] + tensor.dot(input_t, self.S[clk - 1])

        u_on = tensor.nnet.sigmoid(tensor.dot(input_t, self.U_s[clk - 1]) +
                                   tensor.dot(state_below, self.U_x[clk - 1]) +
                                   tensor.dot(state_before[:, cm1:], self.U_h[clk - 1][cm1:, :]) +
                                   self.U_b[clk - 1])

        h = u_on * state_before[:, c0:c1] + (1. - u_on) * tensor.tanh(pre_h)
        z = tensor.set_subtensor(state_before[:, c0:c1], h)

        return h, z


class HierarchicalGatedRecurrent(Recurrent):

    def __init__(self,
                 num_modules=2,
                 update_gate_init_bias=0.,
                 reset_gate_init_bias=0.,
                 global_update_gate_init_bias=0.,
                 global_reset_gate_init_bias=0.,
                 **kwargs):
        super(HierarchicalGatedRecurrent, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    def set_input_space(self, space):

        if (not isinstance(space, SequenceSpace) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Recurrent layer needs a SequenceSpace("
                             "VectorSpace) as input but received  %s instead"
                             % (space))
        self.input_space = space

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim)
        else:
            self.output_space = SequenceSpace(VectorSpace(dim=self.dim))

        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        # So far size of each module should be same
        # It's restricted to use same dimension for each module.
        # It should be generalized.
        # We will use transposed order which is different from
        # the original paper but will give same result.
        assert self.dim % self.num_modules == 0
        self.module_dim = self.dim / self.num_modules

        U = np.zeros((self.dim, self.dim), dtype=config.floatX)
        for i in xrange(self.num_modules):
            for j in xrange(self.num_modules):
                if not((j >= i + 2) and (i < self.num_modules - 2)):
                    if False: #self.istdev is not None:
                        pass
                        # u = rng.randn(self.module_dim,
                        #               self.module_dim)
                        # u *= self.istdev
                    else:
                        u = rng.uniform(-self.irange,
                                        self.irange,
                                        (self.module_dim,
                                         self.module_dim))  #  *\
                            # (rng.uniform(0., 1., (self.module_dim,
                            #                       self.module_dim))
                            #  < self.include_prob)
                    if self.svd:
                        u, s, v = np.linalg.svd(u,
                                                full_matrices=True,
                                                compute_uv=True)
                    U[i*self.module_dim:(i+1)*self.module_dim,
                      j*self.module_dim:(j+1)*self.module_dim] = u
        if True: #self.use_bias:
            self.b = sharedX(np.zeros((self.dim,)) + self.init_bias,
                             name=(self.layer_name + '_b'))
        else:
            assert self.b_lr_scale is None
        self.U = sharedX(U, name=(self.layer_name + '_U'))
        nonzero_idx = np.nonzero(U)
        mask_weights = np.zeros(shape=(U.shape), dtype=config.floatX)
        mask_weights[nonzero_idx[0], nonzero_idx[1]] = 1.
        self.mask = sharedX(mask_weights)

        gate_dim = self.input_space.dim + self.module_dim * (self.num_modules - 1)
        self.gdim = []
        self.gdim.append(np.arange(self.input_space.dim))
        start = self.input_space.dim
        for i in xrange(self.num_modules - 1):
            self.gdim.append(np.arange(start, start + self.module_dim))
            start += self.module_dim

        if self.irange is not None:
            #assert self.istdev is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (gate_dim,
                             self.module_dim))#  *\
                # (rng.uniform(0., 1., (gate_dim,
                #                       self.module_dim))
                #  < self.include_prob)
        # elif self.istdev is not None:
        #     W = rng.randn(gate_dim,
        #                   self.module_dim) * self.istdev
        self.W = sharedX(W, name=(self.layer_name + '_W'))
        if self.irange is not None:
            #assert self.istdev is None
            S = rng.uniform(-self.irange,
                            self.irange,
                            (self.num_modules - 1,
                             self.input_space.dim,
                             self.module_dim))#  *\
                # (rng.uniform(0., 1., (self.num_modules - 1,
                #                       self.input_space.dim,
                #                       self.module_dim))
                #  < self.include_prob)
        # elif self.istdev is not None:
        #     S = rng.randn(self.num_modules - 1,
        #                   self.input_space.dim,
        #                   self.module_dim) * self.istdev
        self.S = sharedX(S, name=(self.layer_name + '_S'))

        # Reset gate switch
        R_s = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.num_modules - 1,
                                    self.input_space.dim,
                                    self.module_dim))
        R_x = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (gate_dim,
                                    self.module_dim))
        R_h = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.num_modules,
                                    self.dim,
                                    self.module_dim))
        self.R_b = sharedX(np.zeros((self.num_modules,
                                     self.module_dim)) +
                           self.reset_gate_init_bias,
                           name=(self.layer_name + '_R_b'))
        self.R_s = sharedX(R_s, name=(self.layer_name + '_R_s'))
        self.R_x = sharedX(R_x, name=(self.layer_name + '_R_x'))
        self.R_h = sharedX(R_h, name=(self.layer_name + '_R_h'))
        # Update gate switch
        U_s = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.num_modules - 1,
                                    self.input_space.dim,
                                    self.module_dim))
        U_x = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (gate_dim,
                                    self.module_dim))
        U_h = self.mlp.rng.uniform(-self.irange,
                                   self.irange,
                                   (self.num_modules,
                                    self.dim,
                                    self.module_dim))
        self.U_b = sharedX(np.zeros((self.num_modules,
                                     self.module_dim)) +
                           self.update_gate_init_bias,
                           name=(self.layer_name + '_U_b'))
        self.U_s = sharedX(U_s, name=(self.layer_name + '_U_s'))
        self.U_x = sharedX(U_x, name=(self.layer_name + '_U_x'))
        self.U_h = sharedX(U_h, name=(self.layer_name + '_U_h'))
        # Global reset gate switch
        gR_s = self.mlp.rng.uniform(-self.irange,
                                    self.irange,
                                    (self.num_modules - 1,
                                     self.input_space.dim, 1))
        gR_x = self.mlp.rng.uniform(-self.irange,
                                    self.irange,
                                    (self.num_modules - 1,
                                     self.module_dim, 1))
        gR_h = self.mlp.rng.uniform(-self.irange,
                                    self.irange,
                                    (self.num_modules - 1,
                                     self.dim, 1))
        self.gR_b = sharedX(np.zeros((self.num_modules - 1, 1)) +
                            self.global_reset_gate_init_bias,
                            name=(self.layer_name + '_gR_b'))
        self.gR_s = sharedX(gR_s, name=(self.layer_name + '_gR_s'))
        self.gR_x = sharedX(gR_x, name=(self.layer_name + '_gR_x'))
        self.gR_h = sharedX(gR_h, name=(self.layer_name + '_gR_h'))
        # Global update gate switch
        gU_s = self.mlp.rng.uniform(-self.irange,
                                    self.irange,
                                    (self.num_modules - 1,
                                     self.input_space.dim, 1))
        gU_x = self.mlp.rng.uniform(-self.irange,
                                    self.irange,
                                    (self.num_modules - 1,
                                     self.module_dim, 1))
        gU_h = self.mlp.rng.uniform(-self.irange,
                                    self.irange,
                                    (self.num_modules - 1,
                                     self.dim, 1))
        self.gU_b = sharedX(np.zeros((self.num_modules - 1, 1)) +
                            self.global_update_gate_init_bias,
                            name=(self.layer_name + '_gU_b'))
        self.gU_s = sharedX(gU_s, name=(self.layer_name + '_gU_s'))
        self.gU_x = sharedX(gU_x, name=(self.layer_name + '_gU_x'))
        self.gU_h = sharedX(gU_h, name=(self.layer_name + '_gU_h'))
        self._params = [self.U, self.W, self.b]

    def get_params(self):
        rval = super(HierarchicalGatedRecurrent, self).get_params()
        rval += [self.R_s, self.R_x, self.R_h, self.R_b]
        rval += [self.gR_s, self.gR_x, self.gR_h, self.gR_b]
        rval += [self.U_s, self.U_x, self.U_h, self.U_b]
        rval += [self.gU_s, self.gU_x, self.gU_h, self.gU_b]
        rval += [self.S]

        return rval

    def _modify_updates(self, updates):

        if self.U in updates:
            updates[self.U] = updates[self.U] * self.mask

    def fprop(self, state_below):
        state_below, mask = state_below

        # z0 is the initial hidden state which is (batch size, output dim)
        z0 = tensor.alloc(np.cast[config.floatX](0), state_below.shape[1],
                          self.dim)
        if self.dim == 1:
            # This should fix the bug described in Theano issue #1772
            z0 = tensor.unbroadcast(z0, 1)

        # Later we will add a noise function
        W = self.W
        U = self.U

        def fprop_step(state_below, mask, state_before, W, U):

            r = tensor.dot(state_before, U)
            h, z = self.first_step(state_below, state_before, r, W, 0)
            h, z = self.gated_step(state_below, h, z, r, W, 1)
            h, z = self.gated_step(state_below, h, z, r, W, 2)

            # Only update the state for non-masked data, otherwise
            # just carry on the previous state until the end
            z = mask[:, None] * z + (1 - mask[:, None]) * state_before
            return z

        z, updates = scan(fn=fprop_step,
                          sequences=[state_below, mask],
                          outputs_info=[z0],
                          non_sequences=[W, U])

        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i] for i in self.indices]
            else:
                return z[self.indices[0]]
        else:
            return (z, mask)

    def first_step(self, state_below, state_before, r, W, clk):

        c0 = clk * self.module_dim
        c1 = (clk + 1) * self.module_dim

        r_on = T.nnet.sigmoid(T.dot(state_below,
                                    self.R_x[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +
                              T.dot(state_before, self.R_h[clk]) +
                              self.R_b[clk])
        pre_h = r_on * r[:, c0:c1] +\
            T.dot(state_below, W[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +\
            self.b[c0:c1]

        u_on = T.nnet.sigmoid(T.dot(state_below,
                                    self.U_x[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +
                              T.dot(state_before, self.U_h[clk]) +
                              self.U_b[clk])

        h = u_on * state_before[:, c0:c1] + (1. - u_on) * tensor.tanh(pre_h)
        z = T.set_subtensor(state_before[:, c0:c1], h)

        return h, z

    def gated_step(self, input_t, state_below, state_before, r, W, clk):

        cm1 = (clk - 1) * self.module_dim
        c0 = clk * self.module_dim
        c1 = (clk + 1) * self.module_dim

        r_on = tensor.nnet.sigmoid(tensor.dot(input_t, self.R_s[clk - 1]) +
                                   tensor.dot(state_below,
                                              self.R_x[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +
                                   tensor.dot(state_before[:, cm1:], self.R_h[clk][cm1:, :]) +
                                   self.R_b[clk])
        gr_on = tensor.nnet.sigmoid(tensor.dot(input_t, self.gR_s[clk - 1]) +
                                    tensor.dot(state_below, self.gR_x[clk - 1]) +
                                    tensor.dot(state_before[:, cm1:], self.gR_h[clk - 1][cm1:, :]) +
                                    self.gR_b[clk - 1])
        gr_on = tensor.addbroadcast(gr_on, 1)
        pre_h = gr_on * r_on * r[:, c0:c1] +\
            tensor.dot(state_below, W[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +\
            self.b[c0:c1] + tensor.dot(input_t, self.S[clk - 1])

        u_on = tensor.nnet.sigmoid(tensor.dot(input_t, self.U_s[clk - 1]) +
                                   tensor.dot(state_below,
                                              self.U_x[self.gdim[clk][0]:self.gdim[clk][-1] + 1, :]) +
                                   tensor.dot(state_before[:, cm1:], self.U_h[clk][cm1:, :]) +
                                   self.U_b[clk])

        gu_on = tensor.nnet.sigmoid(tensor.dot(input_t, self.gU_s[clk - 1]) +
                                    tensor.dot(state_below, self.gU_x[clk - 1]) +
                                    tensor.dot(state_before[:, cm1:], self.gU_h[clk - 1][cm1:, :]) +
                                    self.gU_b[clk - 1])
        gu_on = tensor.addbroadcast(gu_on, 1)

        h = u_on * state_before[:, c0:c1] + (1. - u_on) * tensor.tanh(pre_h)
        h = gu_on * state_before[:, c0:c1] + (1. - gu_on) * h
        z = tensor.set_subtensor(state_before[:, c0:c1], h)

        return h, z

class GatedRecurrent(Recurrent):
    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        super(GatedRecurrent, self).set_input_space(space)

        rng = self.mlp.rng
        # Ug is the hidden-to-gate matrix
        Ug = rng.uniform(-self.irange, self.irange, (self.dim, 2 * self.dim))
        # Wg is the input-to-gate matrix
        Wg = rng.uniform(-self.irange, self.irange, (self.input_space.dim, 2 * self.dim))

        self._params += [sharedX(Wg, name=(self.layer_name + '_Wg')),
                        sharedX(Ug, name=(self.layer_name + '_Ug')),
                        sharedX(np.zeros(2 * self.dim),
                                name=self.layer_name + '_bg')]

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        state_below, mask = state_below

        # z0 is the initial hidden state which is (batch size, output dim)
        z0 = tensor.alloc(np.cast[config.floatX](0), state_below.shape[1], self.dim)
        if self.dim == 1:
            # This should fix the bug described in Theano issue #1772
            z0 = tensor.unbroadcast(z0, 1)

        # Later we will add a noise function
        W, U, b, Wg, Ug, bg = self._params

        # It is faster to do the input-to-hidden matrix multiplications
        # outside of scan
        state_below_g = tensor.dot(state_below, Wg) + bg
        state_below = tensor.dot(state_below, W) + b

        if hasattr(self.nonlinearity, 'fprop'):
            nonlinearity = self.nonlinearity.fprop
        else:
            nonlinearity = self.nonlinearity

        def fprop_step(state_below, state_below_g, mask, state_before):
            g = tensor.nnet.sigmoid(state_below_g + 
                                    tensor.dot(state_before, Ug))
            if g.ndim == 3:
                r = g[:,:,:self.dim]
                u = g[:,:,self.dim:]
            else:
                r = g[:,:self.dim]
                u = g[:,self.dim:]

            new_z = nonlinearity(state_below +
                             r * tensor.dot(state_before, U))
            z = (1. - u) * state_before + u * new_z

            # Only update the state for non-masked data, otherwise
            # just carry on the previous state until the end
            z = mask[:, None] * z + (1 - mask[:, None]) * state_before
            return z, g

        z, updates = scan(fn=fprop_step, sequences=[state_below, state_below_g, mask],
                          outputs_info=[z0, None])
        z = z[0]
        self._scan_updates.update(updates)

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i] for i in self.indices]
            else:
                return z[self.indices[0]]
        else:
            return (z, mask)

