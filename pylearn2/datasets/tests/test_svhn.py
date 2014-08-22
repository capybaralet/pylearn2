"""tests for svhn dataset"""
import unittest
import numpy as np
from pylearn2.datasets.svhn import SVHN, SVHN_On_Memory
from pylearn2.space import Conv2DSpace
from pylearn2.testing.skip import skip_if_no_data


def test_SVHN_On_Memory():
    """
    This test loads the SVHN train, test, and valid datasets
    to RAM.  The largest, train, is about 1 GiB.
    """
    data = SVHN_On_Memory(which_set='train')
    data = SVHN_On_Memory(which_set='test')
    data = SVHN_On_Memory(which_set='valid')

def disabled_test_SVHN_On_Memory():
    """
    This test loads the entire SVHN dataset to RAM.  It is about 7.5 GiBs,
    which is too much for some machines, so by default, this test is
    disabled.
    """
    data = SVHN_On_Memory(which_set='extra')
    data = None
    data = SVHN_On_Memory(which_set='train_all')
    data = None
    data = SVHN_On_Memory(which_set='splitted_train')
