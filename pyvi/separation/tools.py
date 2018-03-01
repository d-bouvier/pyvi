# -*- coding: utf-8 -*-
"""
Module for measuring order separation error.

Functions
---------
error_measure :
    Returns the relative error between orders and their estimates.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from ..utilities.mathbox import rms, safe_db


#==============================================================================
# Functions
#==============================================================================

def error_measure(signals_ref, signals_est, db=True):
    """
    Returns the relative error between orders and their estimates.

    This error is computed as the RMS value of the error estimation divided by
    the RMS values of the true orders, for each order.

    Parameters
    ----------
    signals_ref : array_like
        True homogeneous orders.
    signals_est : array_like
        Estimated homogeneous orders.

    Returns
    -------
    error : numpy.ndarray
        List of normalized-RMS error values.
    """

    rms_error = rms(signals_ref - signals_est, axis=1)
    rms_ref = rms(signals_ref, axis=1)
    rms_ref[rms_ref == 0] = 1
    if db:
        return safe_db(rms_error, rms_ref)
    else:
        return rms_error / rms_ref


def create_vandermonde_mixing_mat(factors, N):
    """
    Creates the Vandermonde matrix due to the nonlinear orders homogeneity.

    Parameters
    ----------
    N : int
        Number of orders to separate (truncation order of the Volterra series).
    factors : array_like
        Factors applied to the base signal in order to create the test signals.

    Returns
    -------
    matrix: np.ndarray (of size=(len(factors), N))
        Mixing matrix of the Volterra orders in the output signals.
    """

    return np.vander(factors, N=N+1, increasing=True)[:, 1::]
