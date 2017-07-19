# -*- coding: utf-8 -*-
"""
Module for handling identification error, kernels and volterra basis.

Functions
---------
error_measure :
    Returns the relative error between kernels and their estimates.
nb_coeff_in_kernel :
    Returns the number of coefficient in a kernel.
nb_coeff_in_all_kernels :
    Returns the number of coefficient in all kernels up to a specified order.
vector_to_kernel :
    Rearranges vector of order n Volterra kernel coefficients into tensor.
vector_to_all_kernels :
    Rearranges vector of Volterra kernels coefficients into N tensors.
volterra_basis_by_order :
    Returns a dict gathering Volterra basis matrix for each order.
volterra_basis_by_term :
    Returns a dict gathering Volterra basis matrix for each combinatorial term.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 19 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import itertools as itr
import numpy as np
import scipy.linalg as sc_lin
from ..utilities.mathbox import rms, safe_db, binomial, array_symmetrization


#==============================================================================
# Functions
#==============================================================================

def error_measure(kernels_ref, kernels_est, db=True):
    """
    Returns the relative error between kernels and their estimates.

    This error is computed as the RMS value of the error estimation divided by
    the RMS values of the true kernels, for each order.

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
    for order, kernel_est in sorted(kernels_est.items()):
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


def nb_coeff_in_kernel(M, n, form='sym'):
    """
    Returns the number of coefficient in a kernel.

    Parameters
    ----------
    M : int
        Memory length of the kernel (in samples).
    n : int
        Kernel order.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).

    Returns
    -------
    nb_coeff : int
        The corresponding number of coefficient.
    """

    if form in {'sym', 'symmetric', 'tri', 'triangular'}:
        return binomial(M + n - 1, n)
    else:
        return M**n


def nb_coeff_in_all_kernels(M, N, form='sym'):
    """
    Returns the number of coefficient in all kernels up to a specified order.

    Parameters
    ----------
    M : int
        Memory length of the kernel (in samples).
    N : int
        Highest kernel order.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).

    Returns
    -------
    nb_coeff : int
        The corresponding number of coefficient.
    """

    if form in {'sym', 'symmetric', 'tri', 'triangular'}:
        return binomial(M + N, N) - 1
    else:
        return sum([nb_coeff_in_kernel(M, n, form=form) for n in range(1, N+1)])


def vector_to_kernel(vec_kernel, M, n, form='sym'):
    """
    Rearranges vector of order n Volterra kernel coefficients into tensor.

    Parameters
    ----------
    vec_kernel : numpy.ndarray
        Vector regrouping all symmetric coefficients of a Volterra kernel.
    M : int
        Memory length of the kernel (in samples).
    n : int
        Kernel order.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).

    Returns
    -------
    kernel : numpy.ndarray
        The corresponding Volterra kernel.
    """

    # Check dimension
    length = nb_coeff_in_kernel(M, n, form=form)
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
    Rearranges vector of Volterra kernels coefficients into N tensors.

    Parameters
    ----------
    f : numpy.ndarray
        Vector regrouping all symmetric coefficients of the Volterra kernels.
    M : int
        Memory length of kernels (in samples).
    N : int
        Highest kernel order.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    # Check dimension
    length = nb_coeff_in_all_kernels(M, N, form=form)
    assert f.shape[0] == length, \
           'The vector of Volterra coefficients has wrong length ' + \
           '(got {}, expected {}).'.format(f.shape[0], length)

    # Initialization
    kernels = dict()
    current_ind = 0

    # Loop on all orders of nonlinearity
    for n in range(1, N+1):
        nb_coeff = nb_coeff_in_kernel(M, n, form=form)
        kernels[n] = vector_to_kernel(f[current_ind:current_ind+nb_coeff],
                                      M, n, form=form)
        current_ind += nb_coeff

    return kernels


def volterra_basis_by_order(signal, M, N):
    """
    Returns a dict gathering Volterra basis matrix for each order.

    Parameters
    ----------
    signal : array_like
        Input signal from which to construct the Volterras basis.
    M : int
        Memory length of kernels (in samples).
    N : int
        Highest kernel order.

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionnary of Volterra basis matrix for each order.
    """

    phi = dict()
    phi[1] = sc_lin.toeplitz(signal, np.zeros((1, M)))

    for n in range(2, N+1):
        size = phi[n-1].shape[1]
        temp = phi[n-1][:, :, np.newaxis] * phi[1][:, np.newaxis, :]
        phi[n] = _reshape_and_eliminate_redudancy(temp, M, n, size)

    return phi


def volterra_basis_by_term(signal, M, N):
    """
    Returns a dict gathering Volterra basis matrix for each combinatorial term.

    Parameters
    ----------
    signal : array_like
        Input signal from which to construct the Volterras basis.
    M : int
        Memory length of kernels (in samples).
    N : int
        Highest kernel order.

    Returns
    -------
    kernels : dict((int, int): numpy.ndarray)
        Dictionnary of Volterra basis matrix for each combinatorial term.
    """

    phi = volterra_basis_by_order(signal, M, N)
    for n in range(1, N+1):
        phi[(n, 0)] = phi.pop(n)

    for n in range(2, N+1):
        size = phi[(n-1, 0)].shape[1]

        # Terms 1 <= k < (n+1)//2
        for k in range(1, (n+1)//2):
            temp = (phi[(n-1, k-1)][:, :, np.newaxis] * \
                    phi[(1, 0)][:, np.newaxis, :].conj() ) + \
                   (phi[(n-1, k)][:, :, np.newaxis] * \
                    phi[(1, 0)][:, np.newaxis, :] )
            phi[(n, k)] = _reshape_and_eliminate_redudancy(temp, M, n, size)
        # Terms k = n//2
        if not n%2:
            temp = 2* np.real(phi[(n-1, n//2-1)][:, :, np.newaxis] * \
                              phi[(1, 0)][:, np.newaxis, :].conj())
            phi[(n, n//2)] = _reshape_and_eliminate_redudancy(temp, M, n, size)

    for (n, k) in phi.keys():
        phi[(n, k)] /= binomial(n, k)

    return phi


def _reshape_and_eliminate_redudancy(array, M, n, size):
    """
    Returns only non-redundant terms of the Volterra basis in matrix form.

    Parameters
    ----------
    array : numpy.ndarray
        3D-array regrouping all terms of the Volterra basis.
    M : int
        Memory length of kernels (in samples).
    n : int
        Current order.
    size : int
        Length of the second dimension of ``array``.

    Returns
    -------
    kernels : numpy.ndarray
        Volterra basis matrix containing only non-dedundant terms.
    """

    temp_tuple = ()
    for ind in range(M):
        idx = int(binomial(n+M-ind-2, n-1))
        temp_tuple += (array[:, size-idx:, ind],)
    return np.concatenate(temp_tuple, axis=1)