# -*- coding: utf-8 -*-
"""
Tools for kernel identification.

Functions
---------
_solver :
    Solve Ax=y using specified method if A is not an empty array.
_ls_solver :
    Compute least-squares solution of Ax=y.
_qr_solver :
    Compute solution of Ax=y using a QR decomposition of A.
_cplx_to_real :
    Cast a numpy.ndarray of complex type to real type with a specified mode.
compute_combinatorial_basis :
    Creates dictionary of combinatorial basis matrix.

Notes
-----
Developed for Python 3.6
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

__all__ = ['compute_combinatorial_basis']


#==============================================================================
# Importations
#==============================================================================

import warnings
from collections.abc import Sequence
import numpy as np
import scipy.linalg as sc_lin
from ..volterra.combinatorial_basis import (volterra_basis, hammerstein_basis,
                                            projected_volterra_basis,
                                            projected_hammerstein_basis)
from ..volterra.tools import series_nb_coeff
from ..utilities.orthogonal_basis import (_OrthogonalBasis,
                                          is_valid_basis_instance)
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
        or nonlinear interconjugate term.

    Returns
    -------
    dict(int or (int, int): numpy.ndarray)
        Dictionary of combinatorial basis matrix for each order or
        interconjugate term.
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


def _solver(A, y, solver):
    """Solve Ax=y using specified method if A is not an empty array."""

    if A.size:
        if solver in {'LS', 'ls'}:
            return _ls_solver(A, y)
        elif solver in {'QR', 'qr'}:
            return _qr_solver(A, y)
        else:
            message = "Unknown solver {}; available solvers are 'LS' or 'QR'."
            raise ValueError(message.format(solver))
    else:
        return np.zeros((0,))


def _ls_solver(A, y):
    """Compute least-squares solution of Ax=y."""

    x, _, _, _ = sc_lin.lstsq(A, y)
    return x


def _qr_solver(A, y):
    """Compute solution of Ax=y using a QR decomposition of A."""

    q, r = sc_lin.qr(A, mode='economic')
    z = np.dot(q.T, y)
    return sc_lin.solve_triangular(r, z)


def _complex2real(sig_cplx, cast_mode='real-imag'):
    """
    Cast a numpy.ndarray of complex type to real type with a specified mode.

    Parameters
    ----------
    sig_cplx : numpy.ndarray
        Array to cast to real numbers.
    cast_mode : {'real', 'imag', 'real-imag'}, optional (default='real-imag')
        Choose how complex number are casted to real numbers.

    Returns
    -------
    numpy.ndarray
        Array `sig_cplx` casted to real numbers following `cast_mode`.
    """

    if cast_mode not in {'real', 'imag', 'real-imag', 'cplx'}:
        warnings.warn("Unknown cast_mode, mode 'real-imag' used.", UserWarning)
        cast_mode = 'real-imag'

    if cast_mode == 'real':
        return np.real(sig_cplx)
    elif cast_mode == 'imag':
        return np.imag(sig_cplx)
    elif cast_mode == 'real-imag':
        return np.concatenate((np.real(sig_cplx), np.imag(sig_cplx)), axis=0)
    elif cast_mode == 'cplx':
        return sig_cplx
