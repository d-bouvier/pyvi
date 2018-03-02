# -*- coding: utf-8 -*-
"""
Module for handling identification error, kernels and volterra basis.

Functions
---------
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

import numpy as np
import scipy.linalg as sc_lin
from ..volterra.tools import kernel_nb_coeff
from ..utilities.mathbox import binomial
from ..utilities.tools import _as_list


#==============================================================================
# Global variables
#==============================================================================

_triangular_strings_opt = {'tri', 'triangular'}
_symmetric_strings_opt = {'sym', 'symmetric'}
_tri_sym_strings_opt = _triangular_strings_opt | _symmetric_strings_opt


#==============================================================================
# Functions
#==============================================================================


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
        nb_coeff = kernel_nb_coeff(m, n, form='tri')
        nb_coeff_bis = kernel_nb_coeff(m_bis, n, form='tri')
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
