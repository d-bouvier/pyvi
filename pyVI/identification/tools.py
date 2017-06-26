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


def vector_to_kernel(vec_kernel, M, n, form='sym'):
    """
    Rearrange a vector containing the coefficients of a Volterra kernel of order
    ``n`` into a numpy.ndarray representing the Volterra kernel.

    Parameters
    ----------
    vec_kernel : numpy.ndarray
        Vector regrouping all symmetric coefficients of a Volterra kernel.
    M : int
        Memory length of the kernel.
    n : int
        Kernel order.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular)

    Returns
    -------
    kernel : numpy.ndarray
        The corresponding Volterra kernel.
    """
    #TODO update docstring

    # Check dimension
    length = binomial(M + n - 1, n)
    assert len(vec_kernel) == length, 'The vector of coefficients for ' + \
            'Volterra kernel of order {} has wrong length'.format(n) + \
            '(got {}, expected {}).'.format(vec_kernel.shape[0], length)

    # Initialization
    kernel = np.zeros((M,)*n, dtype=vec_kernel.dtype)
    current_ind = 0

    # Loop on all combinations for order n
    for indexes in itr.combinations_with_replacement(range(M), n):
        kernel[indexes] = vec_kernel[current_ind]
        current_ind += 1

    if form in {'sym', 'symmetric'}:
        return array_symmetrization(kernel)
    elif form in {'tri', 'triangular'}:
        return kernel


def vector_to_all_kernels(f, M, N, form='sym'):
    """
    Rearrange a numpy vector containing the coefficients of all Volterra kernels
    up to order ``order_max`` into a dictionnary regrouping numpy.ndarray
    representing the Volterra kernels.

    Parameters
    ----------
    f : numpy.ndarray
        Vector regrouping all symmetric coefficients of the Volterra kernels.
    M : int
        Memory length of kernels.
    N : int, optional
        Highest kernel order.

    Returns
    -------
    kernels : dict of numpy.ndarray
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """
    #TODO update docstring

    # Check dimension
    length = binomial(M + N, N) - 1
    assert f.shape[0] == length, \
           'The vector of Volterra coefficients has wrong length ' + \
           '(got {}, expected {}).'.format(f.shape[0], length)

    # Initialization
    kernels = dict()
    current_ind = 0

    # Loop on all orders of nonlinearity
    for n in range(1, N+1):
        nb_term = binomial(M + n - 1, n)
        kernels[n] = vector_to_kernel(f[current_ind:current_ind+nb_term],
                                      M, n, form=form)
        current_ind += nb_term

    return kernels


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