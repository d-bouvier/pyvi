# -*- coding: utf-8 -*-
"""
Tools for order separation.

Functions
---------
_create_vandermonde_mixing_mat :
    Creates a Vandermonde matrix.

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
