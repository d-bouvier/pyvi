# -*- coding: utf-8 -*-
"""
Toolbox for Volterra series system identification.

This package creates identification methods for Volterra kernels. It relies
on a matrix representation of the input-to-output relation of a Volterra
series, and uses linear algebra tools to estimate the kernels coefficients.

Functions
---------
direct_method :
    Direct kernel identification on the output signal.
order_method :
    Separate kernel identification on each nonlinear homogeneous order.
term_method :
    Separate kernel identification on each nonlinear combinatorial term.
iter_method :
    Recursive kernel identification on homophase signals.
Following functions are wrappers for the previous ones, with parameters
'solver' set to 'QR' and 'out_form' to 'sym'.
KLS :
    Kernel identification via Least-Squares method using a QR decomposition.
orderKLS :
    Performs KLS method on each nonlinear homogeneous order.
termKLS :
    Performs KLS method on each combinatorial term.
iterKLS :
    Performs KLS method recursively on homophase signals.

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
from .tools import _solver, _complex2real
from ..volterra.combinatorics import volterra_basis
from ..volterra.tools import series_nb_coeff, vec2series, vec2dict_of_vec
from ..utilities.mathbox import binomial


#==============================================================================
# Functions
#==============================================================================

def direct_method(input_sig, output_sig, M, N, **kwargs):
    """
    Direct kernel identification on the output signal.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_sig : numpy.ndarray
        Output signal; should have the same shape as ``input_sig``.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionary of estimated kernels, where each key is the nonlinear order.
    {}
    """

    def required_nb_data_func(list_nb_coeff):
        """Compute the minimum number of data required."""
        return sum(list_nb_coeff)

    def core_func(phi_by_order, out_sig, solver, cast_mode):
        """Core computation of the identification."""
        mat = np.concatenate([val for n, val in sorted(phi_by_order.items())],
                             axis=1)
        kernels_vec = _solver(mat, out_sig, solver)
        return vec2dict_of_vec(kernels_vec, M, N)

    return _identification(input_sig, output_sig, M, N, required_nb_data_func,
                           core_func, 'order', **kwargs)


def order_method(input_sig, output_by_order, M, N, **kwargs):
    """
    Separate kernel identification on each nonlinear homogeneous order.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_by_order : numpy.ndarray
        Nonlinear homogeneous orders of the output signal; the first dimension
        of the array should be of length ``N``, and each slice along this
        dimension should have the same shape as ``input_sig``.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionary of estimated kernels, where each key is the nonlinear order.
    {}
    """

    def required_nb_data_func(list_nb_coeff):
        """Compute the minimum number of data required."""
        return max(list_nb_coeff)

    def core_func(phi_by_order, out_by_order, solver, cast_mode):
        """Core computation of the identification."""
        kernels_vec = dict()
        for n, phi in phi_by_order.items():
            kernels_vec[n] = _solver(phi, out_by_order[n-1], solver)
        return kernels_vec

    return _identification(input_sig, output_by_order, M, N,
                           required_nb_data_func, core_func, 'order', **kwargs)


def term_method(input_sig, output_by_term, M, N, **kwargs):
    """
    Separate kernel identification on each nonlinear combinatorial term.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_by_term : dict((int, int): numpy.ndarray
        Nonlinear combinatorial terms of the output signal; each array
        contained in the dictionar should have the same shape as ``input_sig``;
        the dictionary should contains all keys ``(n, q)`` for
        ``n in range(1, N+1)`` and ``q in range(1+n//2)``.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionary of estimated kernels, where each key is the nonlinear order.
    {}
    """

    def required_nb_data_func(list_nb_coeff):
        """Compute the minimum number of data required."""
        return max(list_nb_coeff / (1+np.arange(1, N+1)//2))

    def core_func(phi_by_term, out_by_term, solver, cast_mode):
        """Core computation of the identification."""

        for (n, k) in phi_by_term.keys():
            phi_by_term[n, k] = _complex2real(phi_by_term[n, k],
                                              cast_mode=cast_mode)
            out_by_term[n, k] = _complex2real(out_by_term[n, k],
                                              cast_mode=cast_mode)

        kernels_vec = dict()
        for n in range(1, N+1):
            k_vec = list(range(1+n//2))
            phi_n = np.concatenate([phi_by_term[(n, k)] for k in k_vec],
                                   axis=0)
            out_n = np.concatenate([out_by_term[(n, k)] for k in k_vec],
                                   axis=0)
            kernels_vec[n] = _solver(phi_n, out_n, solver)

        return kernels_vec

    return _identification(input_sig, output_by_term, M, N,
                           required_nb_data_func, core_func, 'term', **kwargs)


def iter_method(input_sig, output_by_phase, M, N, **kwargs):
    """
    Recursive kernel identification on homophase signals.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_by_phase : numpy.ndarray
        Homophase signals constituting the output signal; the first dimension
        of the array should be of length ``2N+1``, and each slice along this
        dimension should have the same shape as ``input_sig``; homophase
        signals should be order with corresponding phases as follows:
        ``[0, 1, ... N, -N, ..., -1]``.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionary of estimated kernels, where each key is the nonlinear order.
    {}
    """

    def required_nb_data_func(list_nb_coeff):
        """Compute the minimum number of data required."""
        return max(list_nb_coeff)

    def core_func(phi_by_term, out_by_phase, solver, cast_mode):
        """Core computation of the identification."""

        kernels_vec = dict()
        for n in range(N, 0, -1):
            temp_sig = out_by_phase[n].copy()

            for n2 in range(n+2, N+1, 2):
                k = (n2-n)//2
                temp_sig -= binomial(n2, k) * np.dot(phi_by_term[(n2, k)],
                                                     kernels_vec[n2])

            kernels_vec[n] = _solver(
                _complex2real(phi_by_term[(n, 0)], cast_mode=cast_mode),
                _complex2real(temp_sig, cast_mode=cast_mode), solver)

        return kernels_vec

    return _identification(input_sig, output_by_phase, M, N,
                           required_nb_data_func, core_func, 'term', **kwargs)


def _identification(input_data, output_data, M, N, required_nb_data_func,
                    core_func, sorted_by, solver='LS', out_form='vec',
                    projection=None, phi=None, cast_mode='real-imag'):
    """Core function for kernel identification in linear algebra formalism."""

    # Check that there is enough data to do the identification
    nb_data = input_data.size
    list_nb_coeff = series_nb_coeff(M, N, form='tri', out_by_order=True)
    required_nb_data = required_nb_data_func(list_nb_coeff)
    if nb_data < required_nb_data:
        raise ValueError('Input signal has {} data samples'.format(nb_data) +
                         ', it should have at least ' +
                         '{}.'.format(required_nb_data))

    # Create dictionary of combinatorial matrix
    if phi is None:
        phi = volterra_basis(input_data, M, N, sorted_by=sorted_by)
    else:
        pass
        #TODO check correct

    # Estimate kernels
    kernels_vec = core_func(phi, output_data, solver, cast_mode)

    # Output
    if out_form == 'vec':
        return kernels_vec
    else:
        return vec2series(kernels_vec, M, N, form=out_form)


#========================================#

def KLS(input_sig, output_sig, M, N, **kwargs):
    """
    Kernel identification via Least-Squares method using a QR decomposition.
    """

    kwargs['solver'] = 'QR'
    kwargs['out_form'] = 'sym'
    return direct_method(input_sig, output_sig, M, N, **kwargs)


def orderKLS(input_sig, output_by_order, M, N, **kwargs):
    """
    Performs KLS method on each nonlinear homogeneous order.
    """

    kwargs['solver'] = 'QR'
    kwargs['out_form'] = 'sym'
    return order_method(input_sig, output_by_order, M, N, **kwargs)


def termKLS(input_sig, output_by_term, M, N, **kwargs):
    """
    Performs KLS method on each combinatorial term.
    """

    kwargs['solver'] = 'QR'
    kwargs['out_form'] = 'sym'
    return term_method(input_sig, output_by_term, M, N, **kwargs)


def iterKLS(input_sig, output_by_phase, M, N, **kwargs):
    """
    Performs KLS method recursively on homophase signals.
    """

    kwargs['solver'] = 'QR'
    kwargs['out_form'] = 'sym'
    return iter_method(input_sig, output_by_phase, M, N, **kwargs)


#========================================#

kwargs_docstring_common = """
    Other parameters
    ----------------
    solver : {'LS', 'QR'}, optional (default='LS')
        Method used for solving linear systems; if set to 'LS', a standard
        Least-Squares estimate is used; if set to 'QR', a QR decomposition of
        the matrix to invert is used.
    out_form : {'vec', sym', 'tri'}, optional (default='vec')
        Form under which the kernels are returned; if set to 'vec', only
        vectors regrouping the nonzero coefficients of the diagonal form are
        returned; if set to 'tri' or 'sym', tensors depicting the triangular
        or symmetric form are returned.
    projection : None
        Orthogonal basis unto which the kernel should be projected for
        identification; unused for the moment."""
kwargs_docstring_phi_order = """
    phi : dict(int: numpy.ndarray), optional (default=None)
        Pre-computed dictionary of the combinatorial matrix for each nonlinear
        homogeneous order."""
kwargs_docstring_phi_term = """
    phi : dict((int, int): numpy.ndarray), optional (default=None)
        Pre-computed dictionary of the combinatorial matrix for each nonlinear
        combinatorial term."""
kwargs_docstring_cast_mode = """
    cast_mode : {'real', 'imag', 'real-imag'}, optional (default='real-imag')
        Choose how complex number are casted to real numbers; if set to
        'real-imag', arrays for the real and imaginary part will be stacked."""

for mode in ('direct', 'order', 'term', 'iter'):
    method = locals()[mode + '_method']
    kwargs_docstring = kwargs_docstring_common
    if mode in {'direct', 'order'}:
        kwargs_docstring += kwargs_docstring_phi_order
    elif mode in {'term', 'iter'}:
        kwargs_docstring += kwargs_docstring_phi_term
        kwargs_docstring += kwargs_docstring_cast_mode
    method.__doc__ = method.__doc__.format(kwargs_docstring)

_wrapper_doc_pre = """
    This function is only a wrapper kept for convenience; refer to
    `pyvi.identification.{}_method`.
    """
_wrapper_doc_post = """
    See also
    --------
    pyvi.identification.{}_method
    """

for method, mode in zip((KLS, orderKLS, termKLS, iterKLS),
                        ('direct', 'order', 'term', 'iter')):
    method.__doc__ += _wrapper_doc_pre.format(mode)
    corresponding_method_doc = locals()[mode + '_method'].__doc__
    method.__doc__ += '\n'.join(corresponding_method_doc.splitlines()[2:])
    method.__doc__ += _wrapper_doc_post.format(mode)

del (kwargs_docstring_common, kwargs_docstring_phi_order,
     kwargs_docstring_phi_term, kwargs_docstring_cast_mode, kwargs_docstring,
     method, corresponding_method_doc, mode)
