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
assert_enough_data_samples :
    Assert that there is enough data samples for the identification.
vector_to_kernel :
    Rearranges vector of order n Volterra kernel coefficients into tensor.
kernel_to_vector :
    Rearranges a Volterra kernel in vector form.
vector_to_all_kernels :
    Rearranges vector of Volterra kernels coefficients into N tensors.
volterra_basis :
    Returns a dict gathering Volterra basis matrix for each order or
    combinatorial term.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import itertools as itr
import numpy as np
import scipy.linalg as sc_lin
from ..utilities.mathbox import (rms, safe_db, binomial, multinomial,
                                 array_symmetrization)


#==============================================================================
# Global variables
#==============================================================================

_triangular_strings_opt = {'tri', 'triangular'}
_symmetric_strings_opt = {'sym', 'symmetric'}
_tri_sym_strings_opt = _triangular_strings_opt | _symmetric_strings_opt


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
            if rms_ref == 0:
                rms_ref = 1
        else:
            rms_error = rms(kernel_est)
            rms_ref = 1

        if db:
            errors.append(safe_db(rms_error, rms_ref))
        else:
            errors.append(rms_error / rms_ref)

    return errors


def _as_list(M, N):
    """
    Check that M is a list, or if int returns a list of this length N .

    Parameters
    ----------
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.

    Returns
    -------
    list : list(int)
        List of memory length by order (in samples).
    """

    if isinstance(M, int):
        return [M, ]*N
    elif isinstance(M, list):
        if len(M) != N:
            raise ValueError('M has length {}, but '.format(len(M)) +
                             'truncation order N is {}'.format(N))
        else:
            return M
    else:
        raise ValueError('M is of type {}, '.format(type(M)) +
                         'should be an int or a list of int.')


def nb_coeff_in_kernel(m, n, form='sym'):
    """
    Returns the number of coefficient in a kernel.

    Parameters
    ----------
    m : int
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

    if form in _tri_sym_strings_opt:
        return binomial(m + n - 1, n)
    else:
        return m**n


def nb_coeff_in_all_kernels(M, N, form='sym'):
    """
    Returns the number of coefficient in all kernels up to a specified order.

    Parameters
    ----------
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).

    Returns
    -------
    nb_coeff : int
        The corresponding number of coefficient.
    """

    if form in _tri_sym_strings_opt and isinstance(M, int):
        return binomial(M + N, N) - 1
    else:
        M = _as_list(M, N)
        return sum([nb_coeff_in_kernel(m, n+1, form=form)
                    for n, m in enumerate(M)])


def assert_enough_data_samples(nb_data, max_nb_est, M, N, name):
    """
    Assert that there is enough data samples for the identification.

    Parameters
    ----------
    nb_data : int
        Number of data samples in the input signal used for identification.
    max_nb_est : int
        Maximum size of linear problem to solve.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.
    name : str
        Name of the identification method.

    Raises
    ------
    ValueError
        If L is inferior to the number of Volterra coefficients.
    """

    if nb_data < max_nb_est:
        raise ValueError('Input signal has {} data samples'.format(nb_data) +
                         ', it should have at least {} '.format(max_nb_est) +
                         'for a truncation to order {} '.format(N) +
                         'and a {}-samples memory length'.format(M) +
                         'using {} method.'.format(name))


def vector_to_kernel(vec_kernel, m, n, form='sym'):
    """
    Rearranges vector of order n Volterra kernel coefficients into tensor.

    Parameters
    ----------
    vec_kernel : numpy.ndarray
        Vector regrouping all symmetric coefficients of a Volterra kernel.
    m : int
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
    length = nb_coeff_in_kernel(m, n, form=form)
    if len(vec_kernel) != length:
        raise ValueError('The vector of coefficients for Volterra kernel' +
                         'of order {} has wrong length'.format(n) +
                         '(got {}, '.format(vec_kernel.shape[0]) +
                         'expected {}).'.format(length))

    # Initialization
    kernel = np.zeros((m,)*n, dtype=vec_kernel.dtype)
    current_ind = 0

    # Loop on all combinations for order n
    for indexes in itr.combinations_with_replacement(range(m), n):
        kernel[indexes] = vec_kernel[current_ind]
        current_ind += 1

    if form in _symmetric_strings_opt:
        return array_symmetrization(kernel)
    elif form in _triangular_strings_opt:
        return kernel


def kernel_to_vector(kernel, form='sym'):
    """
    Rearranges a Volterra kernel in vector form.

    Parameters
    ----------
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the given Volterra kernel (symmetric or triangular).

    Returns
    -------
    vec_kernel : numpy.ndarray
        The corresponding Volterra kernel in vector form.
    """

    # Check dimension
    n = kernel.ndim
    M = kernel.shape[0]
    length = nb_coeff_in_kernel(M, n)

    # Initialization
    vec_kernel = np.zeros((length), dtype=kernel.dtype)
    current_ind = 0

    # Applying a factor if kernel is in symmetric form
    if form in _symmetric_strings_opt:
        factor = np.zeros(kernel.shape)
        for indexes in itr.combinations_with_replacement(range(M), n):
            k = [indexes.count(x) for x in set(indexes)]
            factor[indexes] = multinomial(len(indexes), k)
        kernel = kernel * factor

    # Loop on all combinations for
    for indexes in itr.combinations_with_replacement(range(M), n):
        vec_kernel[current_ind] = kernel[indexes]
        current_ind += 1

    return vec_kernel


def vector_to_all_kernels(f, M, N, form='sym'):
    """
    Rearranges vector of Volterra kernels coefficients into N tensors.

    Parameters
    ----------
    f : numpy.ndarray or dict(int: numpy.ndarray)
        Vector regrouping all symmetric coefficients of the Volterra kernels,
        or dictionnary with one vector by order.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    M = _as_list(M, N)

    # Check dimension
    if isinstance(f, np.ndarray):
        length = nb_coeff_in_all_kernels(M, N, form=form)
        if f.shape[0] != length:
            raise ValueError('The vector of Volterra coefficients has wrong ' +
                             'length (got {}, '.format(f.shape[0]) +
                             'expected {}).'.format(length))
    elif not isinstance(f, dict):
        raise TypeError('Cannot handle type {} '.format(type(f)) +
                        "for variable f'")

    # Initialization
    kernels = dict()

    if isinstance(f, np.ndarray):
        current_ind = 0
        for m, n in itr.zip_longest(M, range(1, N+1)):
            nb_coeff = nb_coeff_in_kernel(m, n, form=form)
            kernels[n] = vector_to_kernel(f[current_ind:current_ind+nb_coeff],
                                          m, n, form=form)
            current_ind += nb_coeff
    elif isinstance(f, dict):
        for n, f_n in f.items():
            m = M[n-1]
            kernels[n] = vector_to_kernel(f_n, m, n, form=form)


    return kernels


def volterra_basis(signal, M, N, mode):
    """
    Base function for creating dictionnary of Volterra basis matrix.

    Parameters
    ----------
    signal : array_like
        Input signal from which to construct the Volterras basis.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.
    mode : {'order', 'term'}
        Choose if matrices are computed for each order or combinatorial term.

    Returns
    -------
    kernels : dict(int or (int, int): numpy.ndarray)
        Dictionnary of Volterra basis matrix order or combinaorial term.
    """

    M = _as_list(M, N)
    signal = signal.copy()
    len_sig = signal.shape[0]
    signal.shape = (len_sig, 1)

    M_bis = [M[-1]]
    for m in M[-2::-1]:
        M_bis.append(max(m, M_bis[-1]))
    M_bis = M_bis[::-1]

    # Parameters
    if mode == 'order':
        key = 1
    elif mode == 'term':
        key = (1, 0)

    phi = dict()
    phi_bis = dict()
    phi[key] = sc_lin.toeplitz(signal, np.zeros((1, M[0])))
    phi_bis[key] = sc_lin.toeplitz(signal, np.zeros((1, M_bis[0])))

    max_delay = dict()
    for n in range(1, N+1):
        max_delay[n] = dict()
    max_delay[1][0] = np.arange(M_bis[0])

    # Loop on nonlinear orders
    for n, m, m_bis in zip(range(2, N+1), M[1:], M_bis[1:]):
        nb_coeff = nb_coeff_in_kernel(m, n, form='tri')
        nb_coeff_bis = nb_coeff_in_kernel(m_bis, n, form='tri')
        delay = np.concatenate(tuple(max_delay[n-1].values()))
        ind_bis = np.where(delay < m_bis)[0]
        dec_bis = len(ind_bis)
        max_delay[n][0] = delay[ind_bis]
        ind = np.where(max_delay[n][0] < m)[0]
        dec = len(ind)

        # Initialization
        if mode == 'order':
            phi[n] = np.zeros((len_sig, nb_coeff), dtype=signal.dtype)
            phi_bis[n] = np.zeros((len_sig, nb_coeff_bis), dtype=signal.dtype)
        elif mode == 'term':
            for k in range(1 + n//2):
                phi[n, k] = np.zeros((len_sig, nb_coeff), dtype=signal.dtype)
                phi_bis[n, k] = np.zeros((len_sig, nb_coeff_bis),
                                         dtype=signal.dtype)

        # Computation
        if mode == 'order':
            phi_bis[n][:, :dec_bis] = signal * phi_bis[n-1][:, ind_bis]
            phi[n][:, :dec] = phi_bis[n][:, ind]
        elif mode == 'term':
            # Term k = 0
            phi_bis[(n, 0)][:, :dec_bis] = \
                signal * phi_bis[(n-1, 0)][:, ind_bis]
            phi[(n, 0)][:, :dec] = phi_bis[(n, 0)][:, ind]
            # Terms 1 <= k < (n+1)//2
            for k in range(1, (n+1)//2):
                phi_bis[(n, k)][:, :dec_bis] = \
                    signal * phi_bis[(n-1, k)][:, ind_bis] + \
                    signal.conj() * phi_bis[(n-1, k-1)][:, ind_bis]
                phi[(n, k)][:, :dec] = phi_bis[(n, k)][:, ind]
            # Term k = n//2
            if not n % 2:
                phi_bis[(n, n//2)][:, :dec_bis] = 2 * np.real(
                    signal.conj() * phi_bis[(n-1, n//2-1)][:, ind_bis])
                phi[(n, n//2)][:, :dec] = phi_bis[(n, n//2)][:, ind]

        # Copy of identic values
        for offset in range(1, M_bis[n-1]):
            tmp = max_delay[n][0] + offset
            ind = np.where(tmp < m)[0]
            ind_bis = np.where(tmp < m_bis)[0]
            nb_ind = len(ind)
            nb_ind_bis = len(ind_bis)
            max_delay[n][offset] = tmp[ind_bis]

            if mode == 'order':
                phi_bis[n][offset:, dec_bis:dec_bis+nb_ind_bis] = \
                                        phi_bis[n][:-offset, ind_bis]
                phi[n][offset:, dec:dec+nb_ind] = phi_bis[n][:-offset, ind]
            elif mode == 'term':
                for k in range(1 + n//2):
                    phi_bis[(n, k)][offset:, dec_bis:dec_bis+nb_ind_bis] = \
                                        phi_bis[(n, k)][:-offset, ind_bis]
                    phi[(n, k)][offset:, dec:dec+nb_ind] = \
                                        phi[(n, k)][:-offset, ind]
            dec_bis += nb_ind_bis
            dec += nb_ind

    if mode == 'term':
        for (n, k) in phi.keys():
            phi[(n, k)] = phi[(n, k)] / binomial(n, k)

    return phi
