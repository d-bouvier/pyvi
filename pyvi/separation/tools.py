# -*- coding: utf-8 -*-
"""
Tools for order separation.

Functions
---------
_create_vandermonde_mixing_mat :
    Creates a Vandermonde matrix.
_demix_coll :
    Demix a collection of signals via inverse or pseudo-inverse.

Notes
-----
Developed for Python 3.6
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np


#==============================================================================
# Functions
#==============================================================================

def _create_vandermonde_mixing_mat(factors, N, first_column=False):
    """
    Creates the Vandermonde matrix due to the nonlinear orders homogeneity.

    Parameters
    ----------
    factors : array_like
        Factors of the Vandermonde matrix.
    N : int
        Maximum degree of the monomials in the Vandermonde matrix.
    constant_column : boolean, optional (default=False)
        If True, the first column of 1's is kept; otherwise it is discarded.

    Returns
    -------
    numpy.ndarray
        Mixing matrix of the Volterra orders in the output signals; its shape
        verifies ``(len(factors), N)``.
    """

    temp_mat = np.vander(factors, N=N+1, increasing=True)
    if first_column:
        return temp_mat
    else:
        return temp_mat[:, 1::]


def _compute_condition_number(mixing_mat, p=None):
    """
    Compute the condition number of a matrix.

    Parameters
    ----------
    mixing_mat : array_like
        Mixing matrix for which the condition number will be returned.
    p : {None, 1, -1, 2, -2, inf, -inf, 'fro'}, optional
        Order of the norm
        :ref:`(see np.linalg.norm for more details) <np.linalg.norm>`

    Returns
    -------
    c : {float, inf}
        The condition number of the matrix. May be infinite.
    """

    is_square = mixing_mat.shape[0] == mixing_mat.shape[1]
    if is_square or (p is None):
        return np.linalg.cond(mixing_mat, p=p)
    else:
        norm_direct = np.linalg.norm(mixing_mat, p)
        norm_inv = np.linalg.norm(np.linalg.pinv(mixing_mat), p)
        return norm_direct * norm_inv


def _demix_coll(sig_coll, mixing_mat):
    """
    Demix a collection of signals via inverse or pseudo-inverse.

    Parameters
    ----------
    sig_coll : array_like
        Collection of signals; it verifies ``sig_coll.ndim >= 2``.
    mixing_mat : array_like
        Mixing matrix that will be inverted; it verifies
        ``mixing_mat.shape[1] == sig_coll.shape[1]``.

    Returns
    -------
    numpy.ndarray
        Separated signals.
    """

    is_square = mixing_mat.shape[0] == mixing_mat.shape[1]
    if is_square:
        inv_mixing_mat = np.linalg.inv(mixing_mat)
    else:
        inv_mixing_mat = np.linalg.pinv(mixing_mat)
    return np.tensordot(inv_mixing_mat, sig_coll, axes=1)
