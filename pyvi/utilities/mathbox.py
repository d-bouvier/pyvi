# -*- coding: utf-8 -*-
"""
Tooolbox for useful small math functions.

Notes
-----
@author:    bouvier@ircam.fr
            Damien Bouvier, IRCAM, Paris

Last modified on 22 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
#Importations
#==============================================================================

import numpy as np


#==============================================================================
# Functions
#==============================================================================

def rms(sig, axis=None):
    """
    Computation of the root-mean-square of a vector.
    """
    return np.sqrt( np.mean(np.abs(sig)**2, axis=axis) )


def db(val, ref=1):
    """
    Conversion to dB.
    """
    return 20 * np.log10(val / ref)


def safe_db(num, den):
    """
    Conversion to dB with verification that neither the denominator nor
    numerator are equal to zero.
    """

    # Initialization
    if type(num) != np.ndarray:
        _num = np.array(num)
    else:
        _num = num

    if type(den) != np.ndarray:
        _den = np.array(den)
    else:
        _den = den

    # Assert same shape
    assert _num.shape == _den.shape, 'Dimensions of num and den not equal ' + \
            '(they are respectively {} and {}).'.format(_num.shape, _den.shape)

    if _num.shape == ():
        if _num == 0:
            result = np.Inf
        elif _den == 0:
            result = - np.Inf
        else:
            result = db(_num, _den)
    else:
        result = np.zeros(_num.shape)

        # Searching where denominator or numerator is null
        idx_den_null = np.where(_den == 0)
        idx_num_null = np.where(_num == 0)
        idx_not_null = np.ones(_num.shape, np.bool)
        idx_not_null[idx_den_null] = 0
        idx_not_null[idx_num_null] = 0

        # Computation
        result[idx_den_null] = np.Inf
        result[idx_num_null] = - np.Inf
        result[idx_not_null] = db(_num[idx_not_null], _den[idx_not_null])

    return result


def binomial(n, k):
    """
    Binomial coefficient returning an integer.
    """

    from scipy.special import binom as fct_binomial
    return int(fct_binomial(n, k))


def array_symmetrization(array):
    """
    Symmetrize a multidimensional square array (each dimension must have the
    same length).

    Parameters
    ----------
    array : numpy.ndarray
        Array to symmetrize.

    Returns
    -------
    array_sym : numpy.ndarray
        Symmetrized array.
    """

    from math import factorial
    import itertools as itr

    shape = array.shape
    assert len(set(shape)) == 1, 'Multidimensional array is not square ' + \
        '(has shape {})'.format(shape)
    n = len(array.shape)

    array_sym = np.zeros(shape, dtype=array.dtype)
    for ind in itr.permutations(range(n), n):
        array_sym += np.transpose(array, ind)
    return array_sym / factorial(n)
