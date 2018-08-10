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
_cast_sig_complex2real :
    Cast an array of complex type to real type with a specified mode.
_cast_dict_complex2real :
    Cast dictionary of arrays sorted by interconjugate indexes.

Notes
-----
Developed for Python 3.6
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import warnings
import numpy as np
import scipy.linalg as sc_lin


#==============================================================================
# Functions
#==============================================================================

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

    z, R = sc_lin.qr_multiply(A, y, mode='right')
    return sc_lin.solve_triangular(R, z)


def _cast_sig_complex2real(sig_cplx, cast_mode):
    """Cast an array of complex type to real type with a specified mode."""

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


def _cast_dict_complex2real(dict_cplx, cast_mode):
    """Cast dictionary of arrays sorted by interconjugate indexes."""

    casted_dict = dict()
    for (n, k), val in dict_cplx.items():
        if (not n % 2) and (k == n//2):
            casted_dict[(n, k)] = np.real(val)
        else:
            casted_dict[(n, k)] = _cast_sig_complex2real(val, cast_mode)

    return casted_dict
