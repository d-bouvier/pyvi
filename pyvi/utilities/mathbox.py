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
multinomial :
    Multinomial coefficient returning an integer.
array_symmetrization :
    Symmetrize a multidimensional square array.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import math
import itertools as itr
import numpy as np
import scipy.special as sc_sp


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

    return np.sqrt(np.mean(np.abs(sig)**2, axis=axis))


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
    if not isinstance(num, np.ndarray):
        _num = np.array(num)
    else:
        _num = num

    if not isinstance(den, np.ndarray):
        _den = np.array(den)
    else:
        _den = den

    # Broadcast arrays
    _num, _den = np.broadcast_arrays(_num, _den)

    if _num.shape == ():
        if _num == 0:
            result = - np.Inf
        elif _den == 0:
            result = np.Inf
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

    Parameters
    ----------
    n : int
    k : int

    Returns
    -------
    result : int
         Binomial coefficient.
    """

    return sc_sp.comb(n, k, exact=True, repetition=False)


def multinomial(n, k):
    """
    Multinomial coefficient returning an integer.

    Parameters
    ----------
    n : int
    k : list of int

    Returns
    -------
    result : int
         Multinomial coefficient.
    """

    ret = sc_sp.factorial(n)
    for i in k:
        ret //= sc_sp.factorial(i)
    return ret


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
    return array_sym / math.factorial(n)
