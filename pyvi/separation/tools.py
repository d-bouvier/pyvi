# -*- coding: utf-8 -*-
"""
Tools for order separation.

Functions
---------
_create_vandermonde_mixing_mat :
    Creates the Vandermonde matrix due to the nonlinear orders homogeneity.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np


#==============================================================================
# Functions
#==============================================================================

def _create_vandermonde_mixing_mat(factors, N):
    """
    Creates the Vandermonde matrix due to the nonlinear orders homogeneity.

    Parameters
    ----------
    N : int
        Number of orders to separate (truncation order of the Volterra series).
    factors : array_like
        Factors applied to the base signal in order to create the test signals.

    Returns
    -------
    matrix: numpy.ndarray
        Mixing matrix of the Volterra orders in the output signals; its shape
        verifies ``(len(factors), N)``.
    """

    return np.vander(factors, N=N+1, increasing=True)[:, 1::]
