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

import warnings
import numpy as np
import scipy.linalg as sc_lin
from .tools import assert_enough_data_samples, complex2real
from ..volterra.combinatorics import volterra_basis
from ..volterra.tools import series_nb_coeff, vec2dict_of_vec, vec2series
from ..utilities.mathbox import binomial


#==============================================================================
# Functions
#==============================================================================

def _ls_solver(A, y):
    if A.size:
        x, _, _, _ = sc_lin.lstsq(A, y)
        return x
    else:
        return np.zeros((0,))


def _qr_solver(A, y):
    if A.size:
        q, r = sc_lin.qr(A, mode='economic')
        z = np.dot(q.T, y)
        return sc_lin.solve_triangular(r, z)
    else:
        return np.zeros((0,))


def _core_direct_mode(phi_by_order, out_sig, solver_func, N, M):

    mat = np.concatenate([val for n, val in sorted(phi_by_order.items())],
                         axis=1)
    kernels_vec = solver_func(mat, out_sig)
    return vec2dict_of_vec(kernels_vec, M, N)


def _core_order_mode(phi_by_order, out_by_order, solver_func):

    kernels_vec = dict()
    for n, phi in phi_by_order.items():
        kernels_vec[n] = solver_func(phi, out_by_order[n-1])
    return kernels_vec


def _core_term_mode(phi_by_term, out_by_term, solver_func, N, cast_mode):

    for (n, k) in phi_by_term.keys():
        phi_by_term[n, k] = complex2real(phi_by_term[n, k],
                                         cast_mode=cast_mode)
        out_by_term[n, k] = complex2real(out_by_term[n, k],
                                         cast_mode=cast_mode)

    kernels_vec = dict()
    for n in range(1, N+1):
        k_vec = list(range(1+n//2))
        phi_n = np.concatenate([phi_by_term[(n, k)] for k in k_vec], axis=0)
        out_n = np.concatenate([out_by_term[(n, k)] for k in k_vec], axis=0)
        kernels_vec[n] = solver_func(phi_n, out_n)

    return kernels_vec


def _core_iter_mode(phi_by_term, out_by_phase, solver_func, N, cast_mode):

    kernels_vec = dict()

    for n in range(N, 0, -1):
        temp_sig = out_by_phase[n].copy()

        for n2 in range(n+2, N+1, 2):
            k = (n2-n)//2
            temp_sig -= binomial(n2, k) * np.dot(phi_by_term[(n2, k)],
                                                 kernels_vec[n2])

        kernels_vec[n] = solver_func(
            complex2real(phi_by_term[(n, 0)], cast_mode=cast_mode),
            complex2real(temp_sig, cast_mode=cast_mode))

    return kernels_vec


def create_method(N, M=None, mode='direct', solver='LS', out_form='vec',
                  projection=None, cast_mode='real-imag'):

    # Handling of contradictory parameters
    _no_M = M is None
    _no_proj = projection is None
    if _no_M and _no_proj:
        raise ValueError('Neither the kernel memory length (parameter ``M``)' +
                         ' nor an orthogonal basis for projection (parameter' +
                         ' ``projection``) were specified.')
    if not _no_proj:
        if not _no_M:
            message = 'Both parameters ``projection`` and ``M`` were ' + \
                      'specified, which is redundant; parameter ``M`` ' + \
                      'will not be taken into account'
            warnings.warn(message, UserWarning)
            M = None
        if not out_form == 'vec':
            message = "Parameters ``out_form`` was set to '{}', but a " + \
                      "projection is used; thus ``out_form`` is set to 'vec'."
            warnings.warn(message.format(out_form), UserWarning)
            out_form == 'vec'

    # Function for computation of the required minimum number of data samples
    if mode == 'direct':
        _compute_required_nb_data = lambda x: sum(x)
    elif mode in {'order', 'term'}:
        _compute_required_nb_data = lambda x: max(x)
    elif mode == 'iter':
        _compute_required_nb_data = lambda x: max(x / (1+np.arange(1, N+1)//2))

    # Function for the computation of the combinatorial basis
    if _no_proj:
        if mode in {'direct', 'order'}:
            _compute_phi = lambda x: volterra_basis(x, M, N, sorted_by='order')
        elif mode in {'term', 'iter'}:
            _compute_phi = lambda x: volterra_basis(x, M, N, sorted_by='term')
    else:
        raise NotImplementedError

    # Solver method
    if solver == 'LS':
        func_solver = _ls_solver
    elif solver == 'QR':
        func_solver = _qr_solver

    # Function for the core computation
    if mode == 'direct':
        _identification = lambda phi, out, solver: _core_direct_mode(phi, out,
                                                                     solver,
                                                                     N, M)
    elif mode == 'order':
        _identification = _core_order_mode
    elif mode == 'term':
        _identification = lambda phi, out, solver: _core_term_mode(phi, out,
                                                                   solver, N,
                                                                   cast_mode)
    elif mode == 'iter':
        _identification = lambda phi, out, solver: _core_iter_mode(phi, out,
                                                                   solver, N,
                                                                   cast_mode)

    # Function for putting the kernels in the wanted form
    if not _no_proj or out_form == 'vec':
        vec2out = lambda x: x
    else:
        vec2out = lambda x: vec2series(x, M, N, form=out_form)

    # Creation of the identificaiton method function
    def method(input_sig, output, phi=None):

        # Checking that there is enough data samples
        nb_data = input_sig.size
        list_nb_coeff = series_nb_coeff(M, N, form='tri', out_by_order=True)
        required_nb_data = _compute_required_nb_data(list_nb_coeff)
        assert_enough_data_samples(nb_data, required_nb_data, M, N, name=mode)

        # Input combinatoric
        if phi is None:
            phi = _compute_phi(input_sig)

        vec = _identification(phi, output, func_solver)

        return vec2out(vec)

    return method


def KLS(input_sig, output_sig, M, N, phi=None, out_form='sym'):
    """
    Wrapper for identification via Least-Squares using a QR decomposition.

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
    method = create_method(N, M=M, mode='direct', solver='QR',
                           out_form=out_form)
    return method(input_sig, output_sig, phi=phi)


def orderKLS(input_sig, output_by_order, M, N, phi=None, out_form='sym'):
    """
    Wrapper for KLS method on each nonlinear homogeneous order.

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

    method = create_method(N, M=M, mode='order', solver='QR',
                           out_form=out_form)
    return method(input_sig, output_by_order, phi=phi)


def termKLS(input_sig, output_by_term, M, N, phi=None, out_form='sym',
            cast_mode='real-imag'):
    """
    Wrapper for KLS method on each combinatorial term.

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

    method = create_method(N, M=M, mode='term', solver='QR', out_form=out_form,
                           cast_mode=cast_mode)
    return method(input_sig, output_by_term, phi=phi)


def iterKLS(input_sig, output_by_phase, M, N, phi=None, out_form='sym',
            cast_mode='real-imag'):
    """
    Wrapper for KLS method recursively on homogeneous-phase signals.

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

    method = create_method(N, M=M, mode='iter', solver='QR', out_form=out_form,
                           cast_mode=cast_mode)
    return method(input_sig, output_by_phase, phi=phi)
