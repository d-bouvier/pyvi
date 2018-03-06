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
from .tools import (_assert_enough_data_samples, _core_direct_mode,
                    _core_order_mode, _core_term_mode, _core_iter_mode,
                    _ls_solver, _qr_solver)
from ..volterra.combinatorics import volterra_basis
from ..volterra.tools import series_nb_coeff, vec2series


#==============================================================================
# Functions
#==============================================================================

def create_method(N, M=None, mode='direct', solver='LS', out_form='vec',
                  projection=None, cast_mode='real-imag'):
    """
    Factory function for creating Volterra series identification method.

    The identification methods created rely on a matrix representation of the
    input-to-output relation of a Volterra series; it uses linear algebra
    tools to estimate the kernels coefficients.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input signal.
    output_by_phase : numpy.ndarray
        Output signal separated in homogeneous-phase signals.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.
    phi : {None, dict(int: numpy.ndarray)}, optional (default=None)
        If None, ``phi`` is computed from ``input_data``; else, ``phi`` is used.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).
    cast_mode : {'real', 'imag', 'real-imag'}, optional (default='real-imag')
        Choose how complex number are casted to real numbers.

    Returns
    -------
    method : function
        Custom identification method; it accepts input and output data as
        parameters, and returns the estimated kernels.
    """

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

    def method(input_data, output, phi=None):
        # Checking that there is enough data samples
        nb_data = input_data.size
        list_nb_coeff = series_nb_coeff(M, N, form='tri', out_by_order=True)
        required_nb_data = _compute_required_nb_data(list_nb_coeff)
        _assert_enough_data_samples(nb_data, required_nb_data, M, N, name=mode)

        # Input combinatoric
        if phi is None:
            phi = _compute_phi(input_data)

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
