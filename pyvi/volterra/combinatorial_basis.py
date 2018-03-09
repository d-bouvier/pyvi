# -*- coding: utf-8 -*-
"""
Module for computing volterra combinatorial basis.

Functions
---------
compute_combinatorial_basis :
    Creates dictionary of combinatorial basis matrix.
volterra_basis :
    Dictionary of combinatorial basis matrix for Volterra system.
hammerstein_basis :
    Dictionary of combinatorial basis matrix for Hammerstein system.
projected_volterra_basis :
    Dictionary of combinatorial basis matrix for projected Volterra system.
projected_hammerstein_basis :
    Dictionary of combinatorial basis matrix for projected Hammerstein system.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import warnings
import itertools as itr
from collections.abc import Sequence
import numpy as np
import scipy.linalg as sc_lin
from .tools import kernel_nb_coeff, series_nb_coeff
from ..utilities.orthogonal_basis import (_OrthogonalBasis,
                                          is_valid_basis_instance)
from ..utilities.mathbox import binomial
from ..utilities.tools import _as_list


#==============================================================================
# Constants
#==============================================================================

_STRING_VOLTERRA = {'volterra', 'Volterra', 'VOLTERRA'}
_STRING_HAMMERSTEIN = {'hammerstein', 'Hammerstein', 'HAMMERSTEIN'}


#==============================================================================
# Functions
#==============================================================================

def compute_combinatorial_basis(signal, N, system_type='volterra', M=None,
                                orthogonal_basis=None, sorted_by='order'):
    """
    Creates dictionary of combinatorial basis matrix.

    Parameters
    ----------
    signal : array_like
        Input signal from which to construct the Volterras basis.
    N : int
        Truncation order.
    M : int or list(int)
        Memory length for each kernels (in samples).
    system_type : {'volterra', 'hammerstein'}, optional (default='volterra')
        Assumed type of the system; if set to 'volterra', combinatorial basis
        contains all possible input products; if set to 'hammerstein',
        combinatorial basis only contains those corresponding to diagonal
        kernel values.
    orthogonal_basis : (list of) basis object, optional (default=None)
        Orthogonal basis unto which kernels are projected; can be specified
        globally for all orders, or separately for each order via a list of
        different values. See module :mod:`pyvi.utilities.orthogonal_basis`
        for precisions on what basis object can be.
    sorted_by : {'order', 'term'}, optional (default='order')
        Choose if matrices are computed for each nonlinear homogeneous order
        or nonlinear combinatorial term.

    Returns
    -------
    kernels : dict(int or (int, int): numpy.ndarray)
        Dictionary of combinatorial basis matrix for each order or
        combinatorial term.
    """

    _M, orthogonal_basis_is_list = _check_parameters(N, system_type, M,
                                                     orthogonal_basis)

    if orthogonal_basis is None:
        if system_type in _STRING_VOLTERRA:
            return volterra_basis(signal, N, _M, sorted_by=sorted_by)
        elif system_type in _STRING_HAMMERSTEIN:
            return hammerstein_basis(signal, N, _M, sorted_by=sorted_by)
    else:
        if system_type in _STRING_VOLTERRA:
            return projected_volterra_basis(signal, N, orthogonal_basis,
                                            orthogonal_basis_is_list,
                                            sorted_by=sorted_by)
        elif system_type in _STRING_HAMMERSTEIN:
            return projected_hammerstein_basis(signal, N, orthogonal_basis,
                                               sorted_by=sorted_by)


def _check_parameters(N, system_type, M, orthogonal_basis):
    """Check for wrong, contradictory or missing parameters."""

    if system_type not in set.union(_STRING_VOLTERRA, _STRING_HAMMERSTEIN):
        message = "Unknown system type {}; available types are 'volterra' " + \
                  "or 'hammerstein'."
        raise ValueError(message.format(system_type))

    if M is None and orthogonal_basis is None:
        raise ValueError("Either the memory length `M` or parameter " +
                         "`orthogonal_basis` must be specified.")

    if M is not None and orthogonal_basis is not None:
        message = "Both memory length `M` and parameter `orthogonal_basis`" + \
                  " were specified; memory length `M` will not be used."
        warnings.warn(message, UserWarning)
        M = None

    if M is not None and not isinstance(M, int):
        M = _as_list(M, N)
        if not all([isinstance(m, int) for m in M]):
            raise TypeError("Given memory length `M` is neither an " +
                             "integer nor a list of integer.")

    if orthogonal_basis is not None:
        if isinstance(orthogonal_basis, (Sequence, np.ndarray)):
            orthogonal_basis_is_list = True
            orthogonal_basis = _as_list(orthogonal_basis, N)
            _valid = all([isinstance(basis, _OrthogonalBasis) or
                          is_valid_basis_instance(basis)
                          for basis in orthogonal_basis])
        else:
            orthogonal_basis_is_list = False
            _valid = isinstance(orthogonal_basis, _OrthogonalBasis) or \
                     is_valid_basis_instance(orthogonal_basis)
        if not _valid:
            message = "Given parameter `orthogonal_basis` is not valid."
            raise TypeError(message)
    else:
        orthogonal_basis_is_list = None

    return M, orthogonal_basis_is_list


def _compute_list_nb_coeff(N, system_type, M, orthogonal_basis,
                           orthogonal_basis_is_list):
    """Compute the number of element for each order."""

    if M is not None:
        nb_element = _as_list(M, N)
    elif orthogonal_basis_is_list:
        nb_element = [basis.K for basis in orthogonal_basis]
    else:
        nb_element = _as_list(orthogonal_basis.K, N)

    if system_type in _STRING_VOLTERRA:
        return series_nb_coeff(N, nb_element, form='vec', out_by_order=True)
    else:
        return nb_element


def volterra_basis(signal, N, M, sorted_by):
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


def hammerstein_basis(signal, N, M, sorted_by):
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


def projected_volterra_basis(signal, N, orthogonal_basis,
                             orthogonal_basis_is_list, sorted_by):
    """
    Dictionary of combinatorial basis matrix for projected Volterra system.
    """

    phi = dict()
    sig_proj = dict()

    if orthogonal_basis_is_list:
        _orthogonal_basis = _as_list(orthogonal_basis, N)
        phi[(1, 0)] = orthogonal_basis[0].projection(signal).T
        for n in range(2, N+1):
            sig_proj[n] = _orthogonal_basis[n-1].projection(signal).T
        K_list = [basis.K for basis in _orthogonal_basis]
    else:
        phi[(1, 0)] = orthogonal_basis.projection(signal).T
        for n in range(2, N+1):
            sig_proj[n] = phi[(1, 0)]
        K_list = _as_list(orthogonal_basis.K, N)

    list_nb_coeff = series_nb_coeff(N, K_list, form='tri', out_by_order=True)

    for n in range(2, N+1):
        nb_coeff = list_nb_coeff[n-1]
        curr_K = K_list[n-1]
        phi[(n, 0)] = np.zeros(signal.shape + (nb_coeff,), signal.dtype)

        if sorted_by == 'term':
            for k in range(1, 1+n//2):
                phi[(n, k)] = np.zeros(signal.shape + (nb_coeff,),
                                       signal.dtype)

        ind = 0
        for idx in itr.combinations_with_replacement(range(curr_K), n):
            phi[(n, 0)][:, ind] = np.prod(sig_proj[n][:, idx], axis=1)
            if sorted_by == 'term':
                for k in range(1, 1+n//2):
                    tmp = sig_proj[n][:, idx]
                    total = 0
                    for idx_conj in itr.combinations(range(n), k):
                        tmp2 = tmp.copy()
                        tmp2[:, idx_conj] = tmp2[:, idx_conj].conj()
                        phi[(n, k)][:, ind] += np.prod(tmp2, axis=1)
                        total += 1
                    phi[(n, k)][:, ind] /= total
            ind += 1

        if sorted_by == 'term':
            if not n % 2:
                phi[(n, n//2)] = np.real(phi[(n, n//2)])

    if sorted_by == 'order':
        phi = _phi_by_order_post_processing(phi, N)

    return phi


def projected_hammerstein_basis(signal, N, orthogonal_basis, sorted_by):
    """
    Dictionary of combinatorial basis matrix for projected Hammerstein system.
    """

    _orthogonal_basis = _as_list(orthogonal_basis, N)
    signal = signal.copy()

    phi = dict()
    for n, basis in zip(range(1, N+1), _orthogonal_basis):
        phi[(n, 0)] = basis.projection(signal**n).T

        if sorted_by == 'term':
            # Terms 1 <= k < (n+1)//2
            for k in range(1, (n+1)//2):
                tmp = signal**(n-k) * signal.conj()**k
                phi[(n, k)] = basis.projection(tmp).T
            # Term k = n//2
            if not n % 2:
                tmp = np.real(signal * signal.conj())**(n//2)
                phi[(n, n//2)] = basis.projection(tmp).T

    if sorted_by == 'order':
        phi = _phi_by_order_post_processing(phi, N)

    return phi


def _phi_by_order_post_processing(phi, N):
    """Post processing of the dictionary `phi` if it by nonlinear order."""

    for n in range(1, N+1):
        phi[n] = phi.pop((n, 0))

    return phi


def _combinatorial_mat_diag_terms(signal, m):
    """Part of the combinatorial matrix corresponding to diagonal terms."""
    return sc_lin.toeplitz(signal, np.zeros((1, m)))
