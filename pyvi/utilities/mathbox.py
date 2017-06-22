# -*- coding: utf-8 -*-
"""
Tooolbox for useful small math functions.

Notes
-----
@author:    bouvier@ircam.fr
            Damien Bouvier, IRCAM, Paris

Last modified on 24 Apr. 2017
Developed for Python 3.6.1
"""

#==============================================================================
#Importations
#==============================================================================

import numpy as np


#==============================================================================
# Functions
#==============================================================================

def rms(sig, axes=None):
    """
    Computation of the root-mean-square of a vector.
    """
    return np.sqrt( np.mean(np.abs(sig)**2, axes=axes) )


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
    if den == 0:
        return np.Inf
    if num == 0:
        return - np.Inf
    return 20 * np.log10(num / den)


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
