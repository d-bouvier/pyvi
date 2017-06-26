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
KLS :
    Performs KLS method on each combinatorial term.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 23 June 2017
Developed for Python 3.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from scipy.linalg import qr, solve_triangular
from .tools import (volterra_basis_by_order, volterra_basis_by_term,
                    vector_to_kernel, vector_to_all_kernels)
from ..utilities.mathbox import binomial


#==============================================================================
# Functions
#==============================================================================

def KLS(input_sig, output_sig, M, N, phi=None, form='sym'):
    """
    Identify the Volterra kernels of a system from input and output signals.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Vector of input signal.
    output_sig : numpy.ndarray
        Vector of output signal.
    M : int
        Memory length of kernels
    order_max : int, optional
        Highest kernel order (default 1).
    separated_orders : boolean, optional
        If True, ``output_sig`` should contain the separated homogeneous order
        of the output, and the identification will be made for each kernel
        separately.

    Returns
    -------
    kernels : dict of numpy.ndarray
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """
    #TODO update docstring

    # Input combinatoric
    phi = _KLS_construct_phi(input_sig, M, N, phi=phi)

    f = _KLS_core_computation(phi, output_sig)

    # Re-arranging vector f into volterra kernels
    kernels = vector_to_all_kernels(f, M, N, form=form)

    return kernels


def _KLS_construct_phi(signal, M, N, phi=None):
    """
    Parameters
    ----------
    signal : numpy.ndarray
        Vector of input signal.
    M : int
        Memory length of kernels.
    order_max : int, optional
        Highest kernel order (default 1).

    Returns
    -------
    phi : numpy.ndarray
        Matrix containing the expression of the Volterra basis functionals for
        all orders (up to ``order_max``) and all samples of ``signal``.
    """
    #TODO update docstring

    if phi is None:
        phi_dict = volterra_basis_by_order(signal, M, N)
    elif type(phi) == dict:
        phi_dict = phi
    elif type(phi) == np.ndarray:
        return phi
    return np.concatenate([val for n, val in sorted(phi_dict.items())], axis=1)


def _KLS_core_computation(combinatorial_matrix, output_sig):
    """
    """
    #TODO docstring

    # QR decomposition
    q, r = qr(combinatorial_matrix, mode='economic')

    # Projection on combinatorial basis
    y = np.dot(q.T, output_sig)

    # Forward inverse
    return solve_triangular(r, y)


def orderKLS(input_sig, output_sig_by_order, M, N, phi=None, form='sym'):
    """
    Identify the Volterra kernels of a system from input and output signals.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Vector of input signal.
    output_sig : numpy.ndarray
        Vector of output signal.
    M : int
        Memory length of kernels.
    order_max : int, optional
        Highest kernel order (default 1).
    separated_orders : boolean, optional
        If True, ``output_sig`` should contain the separated homogeneous order
        of the output, and the identification will be made for each kernel
        separately.

    Returns
    -------
    kernels : dict of numpy.ndarray
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """
    #TODO update docstring

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
    """
    #TODO docstring

    return volterra_basis_by_order(signal, M, N)

