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


def hammerstein_basis(signal, N, M, sorted_by='order'):
    """
    Dictionary of combinatorial basis matrix for Hammerstein system.
    """

    _M = _as_list(M, N)
    signal = signal.copy()
    len_sig = signal.shape[0]
    signal.shape = (len_sig, 1)

    phi = dict()
    for n, m in zip(range(1, N+1), _M):
        phi[(n, 0)] = _combinatorial_mat_diag_terms(signal**n, m)
        if sorted_by == 'term':
            # Terms 1 <= k < (n+1)//2
            for k in range(1, (n+1)//2):
                tmp = signal**(n-k) * signal.conj()**k
                phi[(n, k)] = _combinatorial_mat_diag_terms(tmp, m)
            # Term k = n//2
            if not n % 2:
                tmp = np.real(signal * signal.conj())**(n//2)
                phi[(n, n//2)] = _combinatorial_mat_diag_terms(tmp, m)

    if sorted_by == 'order':
        phi = _phi_by_order_post_processing(phi, N)

    return phi


def volterra_basis(signal, N, M, sorted_by='order'):
    """
    Dictionary of combinatorial basis matrix for Volterra system.
    """

    def _copy_and_shift_columns(n, m, dec):
        """Create delayed versions of columns by copying and shifting them."""

        slices_list = []
        for offset in range(1, m):
            tmp = max_delay[n][0] + offset
            ind = np.where(tmp < m)[0]
            nb_ind = len(ind)
            max_delay[n][offset] = tmp[ind]
            slices_list.append((slice(offset, None, None),
                                slice(dec, dec+nb_ind, None),
                                slice(None, -offset, None), ind))
            dec += nb_ind

        def _core(tmp_phi):
            for slices in slices_list:
                tmp_phi[slices[0], slices[1]] = tmp_phi[slices[2], slices[3]]
            return tmp_phi

        phi[(n, 0)] = _core(phi[(n, 0)])
        if sorted_by == 'term':
            for k in range(1, 1 + n//2):
                phi[(n, k)] = _core(phi[(n, k)])

    # Initialization
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
    max_delay = dict()
    for n in range(1, N+1):
        max_delay[n] = dict()

    # First nonlinear order
    phi[(1, 0)] = _combinatorial_mat_diag_terms(signal, _M[0])
    max_delay[1][0] = np.arange(_M[0])

    # Loop on other nonlinear orders
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
        _copy_and_shift_columns(n, m, dec)

    # Keep only columns where there is no input delayed more than wanted
    for (n, k), val in phi.items():
        current_max_delay = np.concatenate(tuple(max_delay[n].values()))
        phi[(n, k)] = val[:, np.where(current_max_delay < _M_save[n-1])[0]]

    if sorted_by == 'term':
        for (n, k) in phi.keys():
                phi[(n, k)] = phi[(n, k)] / binomial(n, k)
    elif sorted_by == 'order':
        phi = _phi_by_order_post_processing(phi, N)

    return phi


def _phi_by_order_post_processing(phi, N):
    """Post processing of the dictionary ``phi`` if it by nonlinear order."""

    for n in range(1, N+1):
        phi[n] = phi.pop((n, 0))

    return phi


def _combinatorial_mat_diag_terms(signal, m):
    """Part of the combinatorial matrix corresponding to diagonal terms."""
    return sc_lin.toeplitz(signal, np.zeros((1, m)))
