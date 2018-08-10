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
