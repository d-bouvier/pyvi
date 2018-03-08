# -*- coding: utf-8 -*-
"""
Tools for handling Volterra series.

Functions
---------
kernel_nb_coeff :
    Returns the number of nonzero coefficients in a Volterra kernel.
series_nb_coeff :
    Returns the number of nonzero coefficients in a Volterra series.
vec2kernel :
    Rearranges a vector of Volterra coefficients of order n into a tensor.
vec2series :
    Rearranges a vector of all Volterra coefficients into a dict of tensors.
kernel2vec :
    Rearranges a Volterra kernel in vector form.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import itertools as itr
import numpy as np
from ..utilities.mathbox import binomial, multinomial, array_symmetrization
from ..utilities.tools import _as_list


#==============================================================================
# Variables
#==============================================================================

_STRING_OPT_TRI = {'triangular', 'tri', 'TRI'}
_STRING_OPT_SYM = {'symmetric', 'sym', 'SYM'}


#==============================================================================
# Functions
#==============================================================================

def kernel_nb_coeff(n, m, form=None):
    """
    Returns the number of nonzero coefficients in a Volterra kernel.

    Parameters
    ----------
    n : int
        Kernel order.
    m : int
        Memory length of the kernel (in samples).
    form : {None, 'sym', 'tri'}, optional (default=None)
        Form to assume for the kernel; symmetric or triangular form have the
        same number of coefficients; if None, no specific form is assumed.

    Returns
    -------
    nb_coeff : int
        The number of nonzero coefficients.
    """

    if (form in _STRING_OPT_TRI) or (form in _STRING_OPT_SYM):
        return binomial(m + n - 1, n)
    else:
        return m**n


def series_nb_coeff(N, M, form=None, out_by_order=False):
    """
    Returns the number of nonzero coefficients in a Volterra series.

    Parameters
    ----------
    N : int
        Truncation order.
    M : int or list(int)
        Memory length for each kernels (in samples).
    form : {None, 'sym', 'tri'} or list, optional (default=None)
        Kernel form; symmetric or triangular form have the same number of
        coefficients; if None, no specific form is assumed. Can be specified
        globally for all orders, or separately for each order via a list of
        different values.
    out_by_order : boolean, optional (default=False)
        Specify if the output should be the total number of nonzero
        coefficients (if ``out_by_order``is False), or a list of the number of
        nonzero coefficients for each order.

    Returns
    -------
    nb_coeff : int or list(int)
        The total number of nonzero coefficients (if ``out_by_order``is False),
        or a list of the number of nonzero coefficients for each order.
    """

    _M = _as_list(M, N)
    _form = _as_list(form, N)

    nb_coeff = []
    for n, m, current_form in zip(range(1, N+1), _M, _form):
        nb_coeff.append(kernel_nb_coeff(n, m, form=current_form))

    if out_by_order:
        return nb_coeff
    else:
        return sum(nb_coeff)


def vec2kernel(vec, n, m, form=None):
    """
    Rearranges a vector of Volterra coefficients of order n into a tensor.

    Parameters
    ----------
    vec_kernel : numpy.ndarray
        Vector regrouping all triangular coefficients of the kernel.
    n : int
        Kernel order.
    m : int
        Memory length of the kernel (in samples).
    form : {None, 'sym', 'tri'}, optional (default=None)
        Form of the returned kernel; if None, triangular form is used.

    Returns
    -------
    kernel : numpy.ndarray
        The corresponding kernel in tensor shape.
    """

    # Check dimension
    nb_coeff = kernel_nb_coeff(n, m, form='tri')
    if vec.shape[0] != nb_coeff:
        raise ValueError('The vector of coefficients for Volterra kernel' +
                         'of order {} has wrong length'.format(n) +
                         '(got {}, '.format(vec.shape[0]) +
                         'expected {}).'.format(nb_coeff))

    # Initialization
    kernel = np.zeros((m,)*n, dtype=vec.dtype)
    current_ind = 0

    # Loop on all combinations for order n
    for indexes in itr.combinations_with_replacement(range(m), n):
        kernel[indexes] = vec[current_ind]
        current_ind += 1

    if form in _STRING_OPT_SYM:
        return array_symmetrization(kernel)
    else:
        return kernel


def _vec2dict_of_vec(vec, list_nb_coeff):
    """Cut a vector of all coefficients into vectors for each order."""

    dict_of_vec = dict()
    start = 0
    for ind_n, nb_coeff in enumerate(list_nb_coeff):
        dict_of_vec[ind_n+1] = vec[start:start+nb_coeff]
        start += nb_coeff

    return dict_of_vec


def vec2series(vec, N, M, form=None):
    """
    Rearranges a vector of all Volterra coefficients into a dict of tensors.

    Parameters
    ----------
    vec : numpy.ndarray or dict(int: numpy.ndarray)
        Vector regrouping all triangular coefficients of the Volterra kernels,
        or dictionnary with one vector by order.
    N : int
        Truncation order.
    M : int or list(int)
        Memory length for each kernels (in samples).
    form : {None, 'sym', 'tri'} or list, optional (default=None)
        Form of the returned kernels; if None, triangular form is used. Can be
        specified globally for all orders, or separately for each order via a
        list of different values.

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    _M = _as_list(M, N)
    _form = _as_list(form, N)

    if isinstance(vec, np.ndarray):
        list_nb_coeff = series_nb_coeff(N, _M, form='tri', out_by_order=True)
        dict_of_vec = _vec2dict_of_vec(vec, list_nb_coeff)
    elif isinstance(vec, dict):
        dict_of_vec = vec
    else:
        raise TypeError('Cannot handle type {} '.format(type(vec)) +
                        "for variable f'")

    kernels = dict()
    for n, vec_n in dict_of_vec.items():
        kernels[n] = vec2kernel(vec_n, n, _M[n-1], form=_form[n-1])

    return kernels


def kernel2vec(kernel, form=None):
    """
    Rearranges a Volterra kernel in vector form.

    Parameters
    ----------
    kernel : numpy.ndarray
        The kernel to rearrange; should bbe a sqaure tensor.should
    form : {None, 'sym', 'tri'}, optional (default=None)
        Form of the given kernel; if None, no specific form is assumed.

    Returns
    -------
    vec : numpy.ndarray
        The corresponding Volterra kernel in vector form.
    """

    # Check dimension
    if len(set(kernel.shape)) != 1:
        raise ValueError('The given kernel is not square ' +
                         '(it has shape {})'.format(kernel.shape))

    # Initialization
    n = kernel.ndim
    m = kernel.shape[0]
    nb_coeff = kernel_nb_coeff(n, m, form='tri')
    vec = np.zeros((nb_coeff), dtype=kernel.dtype)

    # Symmetrizing kernel if needed
    if not ((form in _STRING_OPT_TRI) or (form in _STRING_OPT_SYM)):
        kernel = array_symmetrization(kernel)

    # Applying factors if kernel is in symmetric form
    if not (form in _STRING_OPT_TRI):
        factor = np.zeros(kernel.shape)
        for indexes in itr.combinations_with_replacement(range(m), n):
            k = [indexes.count(x) for x in set(indexes)]
            factor[indexes] = multinomial(len(indexes), k)
        kernel = kernel * factor

    current_ind = 0
    for indexes in itr.combinations_with_replacement(range(m), n):
        vec[current_ind] = kernel[indexes]
        current_ind += 1

    return vec

