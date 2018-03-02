# -*- coding: utf-8 -*-
"""
Module for computing volterra combinatorial basis.

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
from .tools import kernel_nb_coeff
from ..utilities.mathbox import binomial
from ..utilities.tools import _as_list


#==============================================================================
# Functions
#==============================================================================

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
    dtype = signal.dtype

    M_bis = [M[-1]]
    for m in M[-2::-1]:
        M_bis.append(max(m, M_bis[-1]))
    M_bis = M_bis[::-1]

    phi = dict()
    phi_bis = dict()
    phi[(1, 0)] = sc_lin.toeplitz(signal, np.zeros((1, M[0])))
    phi_bis[(1, 0)] = sc_lin.toeplitz(signal, np.zeros((1, M_bis[0])))

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
        phi[(n, 0)] = np.zeros((len_sig, nb_coeff), dtype=dtype)
        phi_bis[(n, 0)] = np.zeros((len_sig, nb_coeff_bis), dtype=dtype)
        if mode == 'term':
            for k in range(1, 1 + n//2):
                phi[n, k] = np.zeros((len_sig, nb_coeff), dtype=dtype)
                phi_bis[n, k] = np.zeros((len_sig, nb_coeff_bis), dtype=dtype)

        # Computation
        phi_bis[(n, 0)][:, :dec_bis] = \
            signal * phi_bis[(n-1, 0)][:, ind_bis]
        phi[(n, 0)][:, :dec] = phi_bis[(n, 0)][:, ind]
        if mode == 'term':
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
        max_delay[n], slices_list = _copy_and_shift_params(max_delay[n], m,
                                                           dec)
        _, slices_list_bis = _copy_and_shift_params(max_delay[n], m_bis,
                                                    dec_bis)

        phi[(n, 0)] = _copy_and_shift_columns(phi[(n, 0)], slices_list)
        phi_bis[(n, 0)] = _copy_and_shift_columns(phi_bis[(n, 0)],
                                                  slices_list_bis)
        if mode == 'term':
            for k in range(1, 1 + n//2):
                phi[(n, k)] = _copy_and_shift_columns(phi[(n, k)],
                                                      slices_list)
                phi_bis[(n, k)] = _copy_and_shift_columns(phi_bis[(n, k)],
                                                          slices_list_bis)

    if mode == 'term':
        for (n, k) in phi.keys():
            phi[(n, k)] = phi[(n, k)] / binomial(n, k)
    elif mode == 'order':
        for n in range(1, N+1):
            phi[n] = phi.pop((n, 0))

    return phi


def _copy_and_shift_params(max_delay, m, dec):

    slices_list = []
    for offset in range(1, m):
        tmp = max_delay[0] + offset
        ind = np.where(tmp < m)[0]
        nb_ind = len(ind)
        max_delay[offset] = tmp[ind]
        slices_list.append((slice(offset, None, None),
                            slice(dec, dec+nb_ind, None),
                            slice(None, -offset, None), ind))
        dec += nb_ind

    return max_delay, slices_list


def _copy_and_shift_columns(tmp_phi, slices_list):

    for slices in slices_list:
        tmp_phi[slices[0], slices[1]] = tmp_phi[slices[2], slices[3]]

    return tmp_phi
