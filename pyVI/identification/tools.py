# -*- coding: utf-8 -*-
"""
Tools for measuring identification error.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 23 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
import itertools as itr
from scipy.linalg import toeplitz
from ..utilities.mathbox import rms, safe_db, binomial, array_symmetrization


#==============================================================================
# Functions
#==============================================================================

def error_measure(kernels_ref, kernels_est, db=True):
    """
    Returns the relative error between kernels and their estimates.

    This error is computed as the RMS value of the error estimation divided by
    the RMS values of the trus kernels, for each order.

    Parameters
    ----------
    kernels_ref : dict(int: numpy.ndarray)
        Dictionnary of the true kernels.
    kernels_est : dict(int: numpy.ndarray)
        Dictionnary of the estimated kernels.

    Returns
    -------
    error : list(floats)
        List of normalized-RMS error values.
    """

    # Initialization
    errors = []

    # Loop on all estimated kernels
    for order, kernel_est in kernels_est.items():
        if order in kernels_ref:
            rms_error = rms(kernel_est - kernels_ref[order])
            rms_ref = rms(kernels_ref[order])
        else:
            rms_error = rms(kernel_est)
            rms_ref = 1

        if db:
            errors.append(safe_db(rms_error, rms_ref))
        else:
            errors.append(rms_error / rms_ref)

    return errors


def volterra_basis_by_order(signal, M, N):
    """
    """
    #TODO docstring

    phi = dict()
    phi[1] = toeplitz(signal, np.zeros((1, M)))

    for n in range(2, N+1):
        size = phi[n-1].shape[1]
        temp = phi[n-1][:, :, np.newaxis] * phi[1][:, np.newaxis, :]
        phi[n] = _reshape_and_eliminate_redudancy(temp, M, n, size)

    return phi


def volterra_basis_by_term(signal, M, N):
    """
    """
    #TODO docstring

    phi = volterra_basis_by_order(signal, M, N)
    for n in range(1, N+1):
        phi[(n, 0)] = phi.pop(n)

    for n in range(2, N+1):
        size = phi[(n-1, 0)].shape[1]

        # Terms 1 <= k < (n+1)//2
        for k in range(1, (n+1)//2):
            temp = int(binomial(n-1, k-1)) * \
                        ( phi[(n-1, k-1)][:, :, np.newaxis] * \
                          phi[(1, 0)][:, np.newaxis, :].conj() ) + \
                   int(binomial(n-1, k)) * \
                        ( phi[(n-1, k)][:, :, np.newaxis] * \
                          phi[(1, 0)][:, np.newaxis, :] )
            temp /= int(binomial(n, k))
            phi[(n, k)] = _reshape_and_eliminate_redudancy(temp, M, n, size)
        # Terms k = n//2
        if not n%2:
            temp = np.real(phi[(n-1, n//2-1)][:, :, np.newaxis] * \
                           phi[(1, 0)][:, np.newaxis, :].conj())
            phi[(n, n//2)] = _reshape_and_eliminate_redudancy(temp, M, n, size)

    return phi


def _reshape_and_eliminate_redudancy(matrix, M, n, size):
    """
    M: memory
    n : order
    """
    #TODO docstring

    temp_tuple = ()
    for ind in range(M):
        idx = int(binomial(n+M-ind-2, n-1))
        temp_tuple += (matrix[:, size-idx:, ind],)
    return np.concatenate(temp_tuple, axis=1)