# -*- coding: utf-8 -*-
"""
Toolbox for Volterra series system identification.

This package creates identification methods for Volterra kernels. It relies
on a matrix representation of the input-to-output relation of a Volterra
series, and uses linear algebra tools to estimate the kernels coefficients.

It contains five methods using different type of output data; it also defines
wrappers for the family of KLS methods, where some parameters already fixed
(``solver`` is set to 'QR' and ``out_form`` to 'sym').

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
phase_method :
    Separate kernel identification on odd and even homophase signals.
KLS :
    Kernel identification via Least-Squares method using a QR decomposition.
orderKLS :
    Performs KLS method on each nonlinear homogeneous order.
termKLS :
    Performs KLS method on each combinatorial term.
iterKLS :
    Performs KLS method recursively on homophase signals.
phaseKLS :
    Performs KLS method separately on odd and even homophase signals.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from .tools import _solver, _complex2real
from ..volterra.combinatorial_basis import (compute_combinatorial_basis,
                                            _check_parameters,
                                            _compute_list_nb_coeff)
from ..volterra.tools import vec2series, _vec2dict_of_vec
from ..utilities.mathbox import binomial


#==============================================================================
# Functions
#==============================================================================

def direct_method(input_sig, output_sig, N, **kwargs):
    """
    Direct kernel identification on the output signal.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_sig : numpy.ndarray
        Output signal; should have the same shape as ``input_sig``.
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

    def core_func(phi_by_order, out_sig, solver, sizes=[], **kwargs):
        """Core computation of the identification."""
        mat = np.concatenate([val for n, val in sorted(phi_by_order.items())],
                             axis=1)
        kernels_vec = _solver(mat, out_sig, solver)
        return _vec2dict_of_vec(kernels_vec, sizes)

    return _identification(input_sig, output_sig, N, required_nb_data_func,
                           core_func, 'order', **kwargs)


def order_method(input_sig, output_by_order, N, **kwargs):
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

    def core_func(phi_by_order, out_by_order, solver, **kwargs):
        """Core computation of the identification."""
        kernels_vec = dict()
        for n, phi in phi_by_order.items():
            kernels_vec[n] = _solver(phi, out_by_order[n-1], solver)
        return kernels_vec

    return _identification(input_sig, output_by_order, N,
                           required_nb_data_func, core_func, 'order', **kwargs)


def term_method(input_sig, output_by_term, N, **kwargs):
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

    def core_func(phi_by_term, out_by_term, solver, cast_mode='', **kwargs):
        """Core computation of the identification."""

        kernels_vec = dict()
        _phi_by_term = _cast_complex2real(phi_by_term, cast_mode)
        _out_by_term = _cast_complex2real(out_by_term, cast_mode)

        for n in range(1, N+1):
            k_vec = list(range(1+n//2))
            phi_n = np.concatenate([_phi_by_term[(n, k)] for k in k_vec],
                                   axis=0)
            out_n = np.concatenate([_out_by_term[(n, k)] for k in k_vec],
                                   axis=0)
            kernels_vec[n] = _solver(phi_n, out_n, solver)

        return kernels_vec

    return _identification(input_sig, output_by_term, N,
                           required_nb_data_func, core_func, 'term', **kwargs)


def iter_method(input_sig, output_by_phase, N, **kwargs):
    """
    Recursive kernel identification on homophase signals.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_by_phase : numpy.ndarray
        Homophase signals constituting the output signal; the first dimension
        of the array should be of length ``2N+1`` (if the whole phase spectrum
        is given, in the order ``[0, 1, ... N, -N, ..., -1]``) or ``N+1``
        (if only the null-and-positive phases are given); each slice along
        the first dimension should have the same shape as ``input_sig``.
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

    def core_func(phi_by_term, out_by_phase, solver, cast_mode='', **kwargs):
        """Core computation of the identification."""

        kernels_vec = dict()
        _phi_by_term = _cast_complex2real(phi_by_term, cast_mode)
        _out_by_phase = out_by_phase.copy()

        for n in range(N, 0, -1):
            current_phi = _phi_by_term[(n, 0)]
            current_phase_sig = _complex2real(_out_by_phase[n],
                                              cast_mode=cast_mode)

            if n == 2:
                current_phi = np.concatenate(
                    (current_phi, binomial(n, n//2) * _phi_by_term[(n, n//2)]),
                    axis=0)
                current_phase_sig = np.concatenate(
                    (current_phase_sig, _out_by_phase[0]), axis=0)

            kernels_vec[n] = _solver(current_phi, current_phase_sig, solver)

            for k in range(1, 1+n//2):
                p = n - 2*k
                _out_by_phase[p] -= \
                    binomial(n, k)*np.dot(phi_by_term[(n, k)], kernels_vec[n])
        return kernels_vec

    return _identification(input_sig, output_by_phase, N,
                           required_nb_data_func, core_func, 'term', **kwargs)


def phase_method(input_sig, output_by_phase, N, **kwargs):
    """
    Separate kernel identification on odd and even homophase signals.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_by_phase : numpy.ndarray
        Homophase signals constituting the output signal; the first dimension
        of the array should be of length ``2N+1`` (if the whole phase spectrum
        is given, in the order ``[0, 1, ... N, -N, ..., -1]``) or ``N+1``
        (if only the null-and-positive phases are given); each slice along
        the first dimension should have the same shape as ``input_sig``.
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

    def core_func(phi_by_term, out_by_phase, solver, sizes=[], cast_mode=''):
        """Core computation of the identification."""

        L = out_by_phase.shape[1]
        L = 2*L if cast_mode == 'real-imag' else L
        kernels_vec = dict()
        _phi_by_term = _cast_complex2real(phi_by_term, cast_mode)

        for is_odd in [False, True]:
            curr_phases = range(2-is_odd, N+1, 2)
            curr_y = np.concatenate([_complex2real(out_by_phase[p],
                                                   cast_mode=cast_mode)
                                     for p in curr_phases], axis=0)

            curr_phi = np.bmat(
                [[_phi_by_term.get((p+2*k, k), np.zeros((L, sizes[p+2*k-1]))) *
                  binomial(p+2*k, k) for k in range(1-(p+1)//2, 1+(N-p)//2)]
                 for p in curr_phases])

            if not is_odd:
                curr_y = np.concatenate((out_by_phase[0], curr_y), axis=0)
                n_even = range(2, N+1, 2)
                temp = np.concatenate([_phi_by_term[n, n//2] *
                                       binomial(n, n//2) for n in n_even],
                                      axis=1)
                curr_phi = np.concatenate((temp, curr_phi), axis=0)

            curr_f = _solver(curr_phi, curr_y, solver)

            index = 0
            for n in range(1 if is_odd else 2, N+1, 2):
                nb_term = sizes[n-1]
                kernels_vec[n] = curr_f[index:index+nb_term]
                index += nb_term

        return kernels_vec

    return _identification(input_sig, output_by_phase, N,
                           required_nb_data_func, core_func, 'term', **kwargs)


def _identification(input_data, output_data, N, required_nb_data_func,
                    core_func, sorted_by, solver='LS', out_form='vec', M=None,
                    orthogonal_basis=None, phi=None, cast_mode='real-imag'):
    """Core function for kernel identification in linear algebra formalism."""


    _M, is_orthogonal_basis_as_list = _check_parameters(N, 'volterra', M,
                                                        orthogonal_basis)
    list_nb_coeff = _compute_list_nb_coeff(N, 'volterra', M,
                                           orthogonal_basis,
                                           is_orthogonal_basis_as_list)

    # Check that there is enough data to do the identification
    nb_data = input_data.size
    required_nb_data = required_nb_data_func(list_nb_coeff)
    if nb_data < required_nb_data:
        raise ValueError('Input signal has {} data samples'.format(nb_data) +
                         ', it should have at least ' +
                         '{}.'.format(required_nb_data))

    # Create dictionary of combinatorial matrix
    if phi is None:
        phi = compute_combinatorial_basis(input_data, N, M=_M,
                                          orthogonal_basis=orthogonal_basis,
                                          sorted_by=sorted_by)
    else:
        pass
        #TODO check correct

    # Estimate kernels
    kernels_vec = core_func(phi, output_data, solver, sizes=list_nb_coeff,
                            cast_mode=cast_mode)

    # Output
    if out_form == 'vec':
        return kernels_vec
    else:
        return vec2series(kernels_vec, N, M, form=out_form)


def _cast_complex2real(val_by_term, cast_mode):
    """Cast dictionary of values sorted by term from complex to real. """

    _val_by_term = dict()
    for (n, k) in val_by_term.keys():
        if (not n % 2) and (k == n//2):
            _val_by_term[(n, k)] = val_by_term[(n, k)]
        else:
            _val_by_term[(n, k)] = _complex2real(val_by_term[(n, k)],
                                                 cast_mode=cast_mode)
    return _val_by_term


#========================================#

def _kwargs_for_KLS(**kwargs):
    kwargs['solver'] = 'QR'
    kwargs['out_form'] = 'sym'
    return kwargs


def KLS(input_sig, output_sig, N, **kwargs):
    """
    Kernel identification via Least-Squares using a QR decomposition.
    """

    kwargs = _kwargs_for_KLS(**kwargs)
    return direct_method(input_sig, output_sig, N, **kwargs)


def orderKLS(input_sig, output_by_order, N, **kwargs):
    """
    Performs KLS method on each nonlinear homogeneous order.
    """

    kwargs = _kwargs_for_KLS(**kwargs)
    return order_method(input_sig, output_by_order, N, **kwargs)


def termKLS(input_sig, output_by_term, N, **kwargs):
    """
    Performs KLS method on each combinatorial term.
    """

    kwargs = _kwargs_for_KLS(**kwargs)
    return term_method(input_sig, output_by_term, N, **kwargs)


def iterKLS(input_sig, output_by_phase, N, **kwargs):
    """
    Performs KLS method recursively on homophase signals.
    """

    kwargs = _kwargs_for_KLS(**kwargs)
    return iter_method(input_sig, output_by_phase, N, **kwargs)


def phaseKLS(input_sig, output_by_phase, N, **kwargs):
    """
    Performs KLS method separately on odd and even homophase signals.
    """

    kwargs = _kwargs_for_KLS(**kwargs)
    return phase_method(input_sig, output_by_phase, N, **kwargs)


#========================================#

kwargs_docstring_common_pre = """
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
    M : int or list(int), optional (default=None)
        Memory length for each kernels (in samples); can be specified
        globally for all orders, or separately for each order via a list of
        different values.
    orthogonal_basis : (list of) basis object, optional (default=None)
        Orthogonal basis unto which kernels are projected; can be specified
        globally for all orders, or separately for each order via a list of
        different values. See module :mod:`pyvi.utilities.orthogonal_basis'`
        for precisions on what basis object can be."""
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
kwargs_docstring_common_post = """

    Either the memory length `M` or parameter `orthogonal_basis` must be
    specified; if both are `None`, the method will issue an error; if both are
    given, memory length `M` will not be used.
    """

for mode in ('direct', 'order', 'term', 'iter', 'phase'):
    method = locals()[mode + '_method']
    kwargs_docstring = kwargs_docstring_common_pre
    if mode in {'direct', 'order'}:
        kwargs_docstring += kwargs_docstring_phi_order
    elif mode in {'term', 'iter', 'phase'}:
        kwargs_docstring += kwargs_docstring_phi_term
        kwargs_docstring += kwargs_docstring_cast_mode
    kwargs_docstring += kwargs_docstring_common_post
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

for method, mode in zip((KLS, orderKLS, termKLS, iterKLS, phaseKLS),
                        ('direct', 'order', 'term', 'iter', 'phase')):
    method.__doc__ += _wrapper_doc_pre.format(mode)
    corresponding_method_doc = locals()[mode + '_method'].__doc__
    method.__doc__ += '\n'.join(corresponding_method_doc.splitlines()[2:])
    method.__doc__ += _wrapper_doc_post.format(mode)

del (kwargs_docstring_common_pre, kwargs_docstring_common_post,
     kwargs_docstring_phi_order, kwargs_docstring_phi_term,
     kwargs_docstring_cast_mode, kwargs_docstring, method,
     corresponding_method_doc, mode)
