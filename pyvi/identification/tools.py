# -*- coding: utf-8 -*-
"""
Tools for kernel identification.

Functions
---------
assert_enough_data_samples :
    Assert that there is enough data samples for the identification.
cplx_to_real :
    Cast a complex numpy.ndarray to a specific type using a given mode.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import warnings
import numpy as np
import scipy.linalg as sc_lin
from ..utilities.mathbox import binomial
from ..volterra.tools import vec2dict_of_vec


#==============================================================================
# Functions
#==============================================================================

def _core_direct_mode(phi_by_order, out_sig, solver_func, N, M):
    """Core computation of the identification method using 'direct' mode.

    This auxiliary function does a joint kernel estimation on the overall
    output of the system."""

    mat = np.concatenate([val for n, val in sorted(phi_by_order.items())],
                         axis=1)
    kernels_vec = _solver(mat, out_sig, solver_func)
    return vec2dict_of_vec(kernels_vec, M, N)


def _core_order_mode(phi_by_order, out_by_order, solver_func):
    """Core computation of the identification method using 'iter' mode.

    This auxiliary function does a separate kernel estimation on the nonlinear
    homogeneous order."""

    kernels_vec = dict()
    for n, phi in phi_by_order.items():
        kernels_vec[n] = _solver(phi, out_by_order[n-1], solver_func)
    return kernels_vec


def _core_term_mode(phi_by_term, out_by_term, solver_func, N, cast_mode):
    """Core computation of the identification method using 'iter' mode.

    This auxiliary function does a separate kernel estimation on the nonlinear
    combinatorial terms for each order."""

    for (n, k) in phi_by_term.keys():
        phi_by_term[n, k] = _complex2real(phi_by_term[n, k],
                                          cast_mode=cast_mode)
        out_by_term[n, k] = _complex2real(out_by_term[n, k],
                                          cast_mode=cast_mode)

    kernels_vec = dict()
    for n in range(1, N+1):
        k_vec = list(range(1+n//2))
        phi_n = np.concatenate([phi_by_term[(n, k)] for k in k_vec], axis=0)
        out_n = np.concatenate([out_by_term[(n, k)] for k in k_vec], axis=0)
        kernels_vec[n] = _solver(phi_n, out_n, solver_func)

    return kernels_vec


def _core_iter_mode(phi_by_term, out_by_phase, solver_func, N, cast_mode):
    """Core computation of the identification method using 'iter' mode.

    This auxiliary function does an iterative kernel estimation on the
    homophase signals. The iterative step begins at order ``N`` and ends at
    order 1."""

    kernels_vec = dict()

    for n in range(N, 0, -1):
        temp_sig = out_by_phase[n].copy()

        for n2 in range(n+2, N+1, 2):
            k = (n2-n)//2
            temp_sig -= binomial(n2, k) * np.dot(phi_by_term[(n2, k)],
                                                 kernels_vec[n2])

        kernels_vec[n] = _solver(
            _complex2real(phi_by_term[(n, 0)], cast_mode=cast_mode),
            _complex2real(temp_sig, cast_mode=cast_mode), solver_func)

    return kernels_vec


def _solver(A, y, solver):
    """Solve Ax=y using ``solver`` if A is not an empty matrix."""
    if A.size:
        return solver(A, y)
    else:
        return np.zeros((0,))


def _ls_solver(A, y):
    """Compute least-squares solution of Ax=y."""
    x, _, _, _ = sc_lin.lstsq(A, y)
    return x


def _qr_solver(A, y):
    """Compute solution of Ax=y using a QR decomposition of A."""
    if A.size:
        q, r = sc_lin.qr(A, mode='economic')
        z = np.dot(q.T, y)
        return sc_lin.solve_triangular(r, z)
    else:
        return np.zeros((0,))


#========================================#

def _assert_enough_data_samples(nb_data, max_nb_est, M, N, name):
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
                         "using '{}' mode.".format(name))


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
    sig_casted : numpy.ndarray
        Array ``sig_cplx`` casted to real numbers following ``cast_mode``.
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
