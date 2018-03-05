# -*- coding: utf-8 -*-
"""
Toolbox for nonlinear system identification.

This package creates functions for Volterra kernel identification via
Least-Squares methods usign QR decomposition.

Functions
---------
KLS :
    Kernel identification via Least-Squares method using a QR decomposition.
orderKLS :
    Performs KLS method on each nonlinear homogeneous order.
termKLS :
    Performs KLS method on each combinatorial term.
iterKLS :
    Performs KLS method recursively on homogeneous-phase signals.

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
from .tools import assert_enough_data_samples, complex2real
from ..volterra.combinatorics import volterra_basis
from ..volterra.tools import kernel_nb_coeff, series_nb_coeff, vec2series
from ..utilities.tools import _as_list
from ..utilities.mathbox import binomial


#==============================================================================
# Functions
#==============================================================================

def KLS(input_sig, output_sig, M, N, phi=None, form='sym'):
    """
    Kernel identification via Least-Squares method using a QR decomposition.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_sig : numpy.ndarray
        Output signal.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.
    phi : {dict(int: numpy.ndarray), numpy.ndarray}, optional (default=None)
        If None, ``phi`` is computed from ``input_sig``; else, ``phi`` is used.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    # Checking that there is enough data samples
    _KLS_check_feasability(input_sig.shape[0], M, N, form=form)

    # Input combinatoric
    phi = _KLS_construct_phi(input_sig, M, N, phi=phi)

    # Identification
    f = _KLS_core_computation(phi, output_sig)

    # Re-arranging vector f into volterra kernels
    kernels = vec2series(f, M, N, form=form)

    return kernels


def _KLS_check_feasability(nb_data, M, N, form='sym'):
    """Auxiliary function of KLS() for checking feasability."""

    nb_coeff = series_nb_coeff(M, N, form=form)
    assert_enough_data_samples(nb_data, nb_coeff, M, N, name='KLS')


def _KLS_construct_phi(signal, M, N, phi=None):
    """Auxiliary function of KLS() for Volterra basis computation."""

    if phi is None:
        phi_dict = volterra_basis(signal, M, N, mode='order')
    elif isinstance(phi, dict):
        phi_dict = phi
    elif isinstance(phi, np.ndarray):
        return phi
    return np.concatenate([val for n, val in sorted(phi_dict.items())], axis=1)


def _KLS_core_computation(combinatorial_matrix, output_sig):
    """Auxiliary function of KLS() for the core computation."""

    if combinatorial_matrix.size:
        # QR decomposition
        q, r = sc_lin.qr(combinatorial_matrix, mode='economic')

        # Projection on combinatorial basis
        y = np.dot(q.T, output_sig)

        # Forward inverse
        return sc_lin.solve_triangular(r, y)
    else:
        return np.zeros((0,))


#=============================================#

def orderKLS(input_sig, output_by_order, M, N, phi=None, form='sym'):
    """
    Performs KLS method on each nonlinear homogeneous order.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_by_order : numpy.ndarray
        Output signal separated in ``N`` nonlinear homogeneous orders.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.
    phi : {None, dict(int: numpy.ndarray)}, optional (default=None)
        If None, ``phi`` is computed from ``input_sig``; else, ``phi`` is used.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    # Checking that there is enough data samples
    _orderKLS_check_feasability(input_sig.shape[0], M, N, form=form)

    # Input combinatoric
    if phi is None:
        phi = _orderKLS_construct_phi(input_sig, M, N)

    # Identification on each order
    f = dict()
    for n, phi_n in phi.items():
        f[n] = _orderKLS_core_computation(phi_n, output_by_order[n-1])

    # Re-arranging vector f into volterra kernels
    kernels = vec2series(f, M, N, form=form)

    return kernels


def _orderKLS_check_feasability(nb_data, M, N, form='sym', name='orderKLS'):
    """Auxiliary function of orderKLS() for checking feasability."""

    nb_coeff = 0
    for m, n in zip(_as_list(M, N), range(1, N+1)):
        nb_coeff = max(nb_coeff, kernel_nb_coeff(m, n, form=form))
    assert_enough_data_samples(nb_data, nb_coeff, M, N, name=name)


def _orderKLS_construct_phi(signal, M, N):
    """Auxiliary function of orderKLS() for Volterra basis computation."""

    return volterra_basis(signal, M, N, mode='order')


def _orderKLS_core_computation(combinatorial_matrix, output_sig):
    """Auxiliary function of orderKLS()) for the core computation."""

    return _KLS_core_computation(combinatorial_matrix, output_sig)


#=============================================#

def termKLS(input_sig, output_by_term, M, N, phi=None, form='sym',
            cast_mode='real-imag'):
    """
    Performs KLS method on each combinatorial term.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_by_term : dict((int, int): numpy.ndarray}
        Output signal separated in nonlinear combinatorial terms.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.
    phi : {None, dict(int: numpy.ndarray)}, optional (default=None)
        If None, ``phi`` is computed from ``input_sig``; else, ``phi`` is used.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).
    cast_mode : {'real', 'imag', 'real-imag'}, optional (default='real-imag')
        Choose how complex number are casted to real numbers.

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    # Checking that there is enough data samples
    _termKLS_check_feasability(input_sig.shape[0], M, N, form=form)

    # Input combinatoric
    if phi is None:
        phi = _termKLS_construct_phi(input_sig, M, N)

    # Identification
    f = _termKLS_core_computation(phi, output_by_term, N, cast_mode)

    # Re-arranging vector f into volterra kernels
    kernels = vec2series(f, M, N, form=form)

    return kernels


def _termKLS_check_feasability(nb_data, M, N, form='sym'):
    """Auxiliary function of termKLS() for checking feasability."""

    _orderKLS_check_feasability(nb_data, M, N, form=form, name='termKLS')


def _termKLS_construct_phi(signal, M, N):
    """Auxiliary function of termKLS() for Volterra basis computation."""

    return volterra_basis(signal, M, N, mode='term')


def _termKLS_core_computation(phi, output_by_term, N, cast_mode):
    """Auxiliary function of termKLS() using 'mmse' mode."""

    f = dict()

    # Identification on each combinatorial term
    for n in range(1, N+1):
        phi_n = np.concatenate([phi[(n, k)] for k in range(1+n//2)], axis=0)
        sig_n = np.concatenate([output_by_term[(n, k)] for k in range(1+n//2)],
                                axis=0)
        f[n] = _KLS_core_computation(complex2real(phi_n, cast_mode=cast_mode),
                                     complex2real(sig_n, cast_mode=cast_mode))

    return f


#=============================================#

def iterKLS(input_sig, output_by_phase, M, N, phi=None, form='sym',
            cast_mode='real-imag'):
    """
    Performs KLS method recursively on homogeneous-phase signals.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_by_phase : numpy.ndarray
        Output signal separated in homogeneous-phase signals.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.
    phi : {None, dict(int: numpy.ndarray)}, optional (default=None)
        If None, ``phi`` is computed from ``input_sig``; else, ``phi`` is used.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).
    cast_mode : {'real', 'imag', 'real-imag'}, optional (default='real-imag')
        Choose how complex number are casted to real numbers.

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    # Checking that there is enough data samples
    _iterKLS_check_feasability(input_sig.shape[0], M, N, form=form)

    # Input combinatoric
    if phi is None:
        phi = _iterKLS_construct_phi(input_sig, M, N)

    # Identification recursive on each homogeneous-phase signal
    f = dict()
    for n in range(N, 0, -1):
        temp_sig = output_by_phase[n].copy()
        for n2 in range(n+2, N+1, 2):
            k = (n2-n)//2
            temp_sig -= binomial(n2, k) * np.dot(phi[(n2, k)], f[n2])
        f[n] = _KLS_core_computation(
            complex2real(phi[(n, 0)], cast_mode=cast_mode),
            complex2real(temp_sig, cast_mode=cast_mode))

    # Re-arranging vector f into volterra kernels
    kernels = vec2series(f, M, N, form=form)

    return kernels


def _iterKLS_check_feasability(nb_data, M, N, form='sym'):
    """Auxiliary function of iterKLS() for checking feasability."""

    _orderKLS_check_feasability(nb_data, M, N, form='sym', name='iterKLS')


def _iterKLS_construct_phi(signal, M, N):
    """Auxiliary function of iterKLS() for Volterra basis computation."""

    return _termKLS_construct_phi(signal, M, N)


#=============================================#




