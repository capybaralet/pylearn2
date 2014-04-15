"""
Costs for use with the MLP model class.
"""
__authors__ = 'Vincent Archambault-Bouffard, Ian Goodfellow'
__copyright__ = "Copyright 2013, Universite de Montreal"

from theano import tensor as T

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin, NullDataSpecsMixin
from pylearn2.utils import safe_izip
from pylearn2.space import VectorSpace, CompositeSpace

import numpy as np


class Default(DefaultDataSpecsMixin, Cost):
    """
    The default Cost to use with an MLP.
    It simply calls the MLP's cost_from_X method.
    """

    def __init__(self, MDN=False):
        self.MDN = MDN

    supervised = True

    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        return model.cost_from_X(data,self.MDN)


class WeightDecay(NullDataSpecsMixin, Cost):
    """
    coeff * sum(sqr(weights))

    for each set of weights.

    Parameters
    ----------
    coeffs : list
        One element per layer, specifying the coefficient to multiply
        with the cost defined by the squared L2 norm of the weights for
        each layer.

        Each element may in turn be a list, e.g., for CompositeLayers.
    """

    def __init__(self, coeffs):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        total_cost : theano.gof.Variable
            coeff * sum(sqr(weights))
            added up for each set of weights.
        """
        self.get_data_specs(model)[0].validate(data)

        def wrapped_layer_cost(layer, coef):
            try:
                return layer.get_weight_decay(coeff)
            except NotImplementedError:
                if coef==0.:
                    return 0.
                else:
                    raise NotImplementedError(str(type(layer)) +
                            " does not implement get_weight_decay.")

        layer_costs = [ wrapped_layer_cost(layer, coeff)
            for layer, coeff in safe_izip(model.layers, self.coeffs) ]

        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [ cost for cost in layer_costs if cost != 0.]

        if len(layer_costs) == 0:
            rval =  T.as_tensor_variable(0.)
            rval.name = '0_weight_decay'
            return rval
        else:
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'MLP_WeightDecay'

        assert total_cost.ndim == 0

        total_cost.name = 'weight_decay'

        return total_cost


class L1WeightDecay(NullDataSpecsMixin, Cost):
    """
    coeff * sum(abs(weights))

    for each set of weights.

    Parameters
    ----------
    coeffs : list
        One element per layer, specifying the coefficient to multiply
        with the cost defined by the L1 norm of the weights for each
        layer.

        Each element may in turn be a list, e.g., for CompositeLayers.
    """

    def __init__(self, coeffs):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        total_cost : theano.gof.Variable
            coeff * sum(abs(weights))
            added up for each set of weights.
        """
        self.get_data_specs(model)[0].validate(data)
        layer_costs = [ layer.get_l1_weight_decay(coeff)
            for layer, coeff in safe_izip(model.layers, self.coeffs) ]

        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [ cost for cost in layer_costs if cost != 0.]

        if len(layer_costs) == 0:
            rval =  T.as_tensor_variable(0.)
            rval.name = '0_l1_penalty'
            return rval
        else:
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'MLP_L1Penalty'

        assert total_cost.ndim == 0

        total_cost.name = 'l1_penalty'

        return total_cost


class MDNCost(Cost):
    """Mixture Density Network Cost"""

    supervised = True

    def __init__(self, NADE_trick = False):
        """
        http://papers.nips.cc/paper/
        5060-rnade-the-real-valued-neural-autoregressive-density-estimator.pdf
        bottom pg 3
        """
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        """
        #space, sources = self.get_data_specs(model)
        #space.validate(data)

        #self.cost_from_X_data_specs()[0].validate(data) # DEAL WITH THIS!

        X, Y = data
        Y_hat = model.fprop(X)
        Y_hat2 = Y_hat.dimshuffle(1,2,0,3).flatten(2)
        Y2 = Y#.dimshuffle(1,2,0,3).flatten(2)
        mix_coefficients = T.nnet.softmax(Y_hat2[::3].T).T
        means = Y_hat2[1::3] - Y2
        stds = T.nnet.softmax(Y_hat2[2::3].T).T
        ret = -T.log(mix_coefficients/(2*np.pi)**.5/stds*T.exp(-means**2/2/stds**2))
        return ret.sum(axis=1).mean()


    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(),
                                VectorSpace(dim=model.get_output_space().shape[0])])
        sources = ('features', 'targets')
        return (space, sources)
