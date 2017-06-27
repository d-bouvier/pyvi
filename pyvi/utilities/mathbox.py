# -*- coding: utf-8 -*-
"""
Tooolbox for useful math functions.

Functions
---------
rms :
    Returns the root-mean-square along given axis.
db :
    Returns the dB value.
safe_db :
    Returns the dB value, with safeguards if numerator or denominator is null.
binomial :
    Binomial coefficient returning an integer.
array_symmetrization :
    Symmetrize a multidimensional square array.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 28 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
#Importations
#==============================================================================

import numpy as np
import itertools as itr
from math import factorial
from scipy.special import binom as fct_binomial


#==============================================================================
# Functions
#==============================================================================

def rms(sig, axis=None):
    """
    Returns the root-mean-square along given axis.

    Parameters
    ----------
    sig : numpy.ndarray
        Array for which RMS is computed.
    axis : {None, int}, optional (default=None)
        Axis along which the RMS is computed. The default is to compute the
        RMS value of the flattened array.

    Returns
    -------
    rms_value : numpy.ndarray or numpy.float
         Root-mean-square value alogn given ``axis``.
    """

    return np.sqrt( np.mean(np.abs(sig)**2, axis=axis) )


def db(val, ref=1.):
    """
    Returns the dB value.

    Parameters
    ----------
    val : numpy.ndarray or float
        Value for which dB value is wanted.
    ref : float, optiona (default=1.)
        Reference used for the dB computation.

    Returns
    -------
    db_value : numpy.ndarray or numpy.float
         dB value.
    """

    return 20 * np.log10(val / ref)


def safe_db(num, den):
    """
    Returns the dB value, with safeguards if numerator or denominator is null.

    Parameters
    ----------
    num : array_like
        Numerator.
    ref : array_like
        Denominator.

    Returns
    -------
    result : numpy.ndarray
         dB value. Is numpy.Inf if ``num`` == 0 and -numpy.Inf if ``den`` == 0.
    """

    # Initialization
    assert num.shape == den.shape, 'Dimensions of num and den not equal.'
    result = np.zeros(num.shape)

    # Searching where denominator or numerator is null
    idx_den_null = np.where(den == 0)
    idx_num_null = np.where(num == 0)
    idx_not_null = np.ones(num.shape, np.bool)
    idx_not_null[idx_den_null] = 0
    idx_not_null[idx_num_null] = 0

    # Computation
    result[idx_den_null] = np.Inf
    result[idx_num_null] = - np.Inf
    result[idx_not_null] = db(num[idx_not_null], den[idx_not_null])

    return result


def binomial(n, k):
    """
    Binomial coefficient returning an integer.

    Parameters
    ----------
    n : int
    k : int

    Returns
    -------
    result : int
         Binomial coefficient.
    """

    return int(fct_binomial(n, k))


def array_symmetrization(array):
    """
    Symmetrize a multidimensional square array.

    Parameters
    ----------
    array : numpy.ndarray
        Array to symmetrize (each dimension must have the same length).

    Returns
    -------
    array_sym : numpy.ndarray
        Symmetrized array.
    """

    shape = array.shape
    assert len(set(shape)) == 1, 'Multidimensional array is not square ' + \
        '(has shape {})'.format(shape)
    n = len(array.shape)

    array_sym = np.zeros(shape, dtype=array.dtype)
    for ind in itr.permutations(range(n), n):
        array_sym += np.transpose(array, ind)
    return array_sym / factorial(n)
