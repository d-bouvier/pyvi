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
_KLS_construct_phi :
    Auxiliary function of KLS method for Volterra basis computation.
_KLS_core_computation( :
    Auxiliary function of KLS method for the core computation.
_orderKLS_construct_phi :
    Auxiliary function of orderKLS method for Volterra basis computation.
_termKLS_construct_phi :
    Auxiliary function of termKLS method for Volterra basis computation.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 12 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from scipy.linalg import qr, solve_triangular
from .tools import (volterra_basis_by_order, volterra_basis_by_term,
                    nb_coeff_in_kernel, vector_to_kernel, vector_to_all_kernels)
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
    M : int
        Memory length of kernels (in samples).
    N : int
        Highest kernel order.
    phi : {dict(int: numpy.ndarray), numpy.ndarray}, optional (default=None)
        If None, ``phi`` is computed from ``input_sig``; else, ``phi`` is used.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    # Input combinatoric
    phi = _KLS_construct_phi(input_sig, M, N, phi=phi)

    f = _KLS_core_computation(phi, output_sig)

    # Re-arranging vector f into volterra kernels
    kernels = vector_to_all_kernels(f, M, N, form=form)

    return kernels


def _KLS_construct_phi(signal, M, N, phi=None):
    """
    Auxiliary function of KLS method for Volterra basis computation.
    """

    if phi is None:
        phi_dict = volterra_basis_by_order(signal, M, N)
    elif type(phi) == dict:
        phi_dict = phi
    elif type(phi) == np.ndarray:
        return phi
    return np.concatenate([val for n, val in sorted(phi_dict.items())], axis=1)


def _KLS_core_computation(combinatorial_matrix, output_sig):
    """
    Auxiliary function of KLS method for the core computation.
    """

    # QR decomposition
    q, r = qr(combinatorial_matrix, mode='economic')

    # Projection on combinatorial basis
    y = np.dot(q.T, output_sig)

    # Forward inverse
    return solve_triangular(r, y)


def orderKLS(input_sig, output_sig_by_order, M, N, phi=None, form='sym'):
    """
    Performs KLS method on each nonlinear homogeneous order.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_sig_by_order : numpy.ndarray
        Output signal separated in ``N`` nonlinear homogeneous orders.
    M : int
        Memory length of kernels (in samples).
    N : int
        Highest kernel order.
    phi : {None, dict(int: numpy.ndarray)}, optional (default=None)
        If None, ``phi`` is computed from ``input_sig``; else, ``phi`` is used.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    # Input combinatoric
    if phi is None:
        phi = _orderKLS_construct_phi(input_sig, M, N)

    kernels = dict()

    for n, phi_n in phi.items():
        f_n = _KLS_core_computation(phi_n, output_sig_by_order[n-1])

        # Re-arranging vector f_n into volterra kernel of order n
        kernels[n] = vector_to_kernel(f_n, M, n, form=form)

    return kernels


def _orderKLS_construct_phi(signal, M, N):
    """
    Auxiliary function of orderKLS method for Volterra basis computation.
    """

    return volterra_basis_by_order(signal, M, N)


def termKLS(input_sig, output_sigs_by_term, M, N, phi=None, form='sym',
            only_real_part=False):
    """
    Performs KLS method on each combinatorial term.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    output_sigs_by_term : dict((int, int): numpy.ndarray}
        Output signal separated in nonlinear combinatorial terms.
    M : int
        Memory length of kernels (in samples).
    N : int
        Highest kernel order.
    phi : {None, dict(int: numpy.ndarray)}, optional (default=None)
        If None, ``phi`` is computed from ``input_sig``; else, ``phi`` is used.
    form : {'sym', 'tri', 'symmetric', 'triangular'}, optional (default='sym')
        Form of the returned Volterra kernel (symmetric or triangular).
    only_real_part : boolean, optional (default=False)
        Choose whether real and imaginary parts or only real parts are used.

    Returns
    -------
    kernels : dict(int: numpy.ndarray)
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    # Input combinatoric
    if phi is None:
        phi = _termKLS_construct_phi(input_sig, M, N)

    kernels = dict()
    f = dict()
    for n in range(1, N+1):
        f[n] = np.zeros((nb_coeff_in_kernel(M, n, form=form),))

    for (n, k), phi_nk in phi.items():
        if only_real_part:
            phi = 2 * np.real(phi_nk)
            output_sig = 2 * np.real(output_sigs_by_term[(n, k)])
        else:
            phi = np.concatenate((np.real(phi_nk), np.imag(phi_nk)), axis=0)
            output_sig = np.concatenate((np.real(output_sigs_by_term[(n, k)]),
                                         np.imag(output_sigs_by_term[(n, k)])),
                                        axis=0)
        f[n] += _KLS_core_computation(phi, output_sig)

    # Re-arranging vector f_n into volterra kernel of order n
    for n in range(1, N+1):
        kernels[n] = vector_to_kernel(f[n] / (1+n//2), M, n, form=form)

    return kernels


def _termKLS_construct_phi(signal, M, N):
    """
    Auxiliary function of termKLS method for Volterra basis computation.
    """

    return volterra_basis_by_term(signal, M, N)

