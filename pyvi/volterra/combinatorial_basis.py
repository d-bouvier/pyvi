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

def volterra_basis(signal, N, M, sorted_by='order'):
    """
    Computes dictionary of combinatorial basis matrix for Volterra system.

    Parameters
    ----------
    signal : array_like
        Input signal from which to construct the Volterras basis.
    N : int
        Truncation order.
    M : int or list(int)
        Memory length for each kernels (in samples).
    sorted_by : {'order', 'term'}, optional (default='order')
        Choose if matrices are computed for each nonlinear homogeneous order
        or nonlinear combinatorial term.

    Returns
    -------
    kernels : dict(int or (int, int): numpy.ndarray)
        Dictionary of combinatorial basis matrix for each order or
        combinatorial term.
    """

    _M_save = _as_list(M, N)
    signal = signal.copy()
    len_sig = signal.shape[0]
    signal.shape = (len_sig, 1)
    dtype = signal.dtype

    _M = [_M_save[-1]]
    for m in _M_save[-2::-1]:
        _M.append(max(m, _M[-1]))
    _M = _M[::-1]

    phi = dict()
    phi[(1, 0)] = sc_lin.toeplitz(signal, np.zeros((1, _M[0])))

    max_delay = dict()
    for n in range(1, N+1):
        max_delay[n] = dict()
    max_delay[1][0] = np.arange(_M[0])

    # Loop on nonlinear orders
    for n, m in zip(range(2, N+1), _M[1:]):
        nb_coeff = kernel_nb_coeff(n, m, form='tri')
        delay = np.concatenate(tuple(max_delay[n-1].values()))
        ind = np.where(delay < m)[0]

        max_delay[n][0] = delay[ind]
        dec = len(ind)

        # Initialization
        phi[(n, 0)] = np.zeros((len_sig, nb_coeff), dtype=dtype)
        if sorted_by == 'term':
            for k in range(1, 1 + n//2):
                phi[(n, k)] = np.zeros((len_sig, nb_coeff), dtype=dtype)

        # Computation
        phi[(n, 0)][:, :dec] = signal * phi[(n-1, 0)][:, ind]
        if sorted_by == 'term':
            # Terms 1 <= k < (n+1)//2
            for k in range(1, (n+1)//2):
                phi[(n, k)][:, :dec] = \
                    signal * phi[(n-1, k)][:, ind] + \
                    signal.conj() * phi[(n-1, k-1)][:, ind]
            # Term k = n//2
            if not n % 2:
                phi[(n, n//2)][:, :dec] = 2 * np.real(
                    signal.conj() * phi[(n-1, n//2-1)][:, ind])

        # Copy of identic values
        max_delay[n], slices_list = _copy_and_shift_params(max_delay[n], m,
                                                           dec)
        # print(n, max_delay[n])

        phi[(n, 0)] = _copy_and_shift_columns(phi[(n, 0)], slices_list)
        if sorted_by == 'term':
            for k in range(1, 1 + n//2):
                phi[(n, k)] = _copy_and_shift_columns(phi[(n, k)], slices_list)

    for (n, k), val in phi.items():
        current_max_delay = np.concatenate(tuple(max_delay[n].values()))
        phi[(n, k)] = val[:, np.where(current_max_delay < _M_save[n-1])[0]]

    if sorted_by == 'term':
        for (n, k) in phi.keys():
            phi[(n, k)] = phi[(n, k)] / binomial(n, k)
    elif sorted_by == 'order':
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
