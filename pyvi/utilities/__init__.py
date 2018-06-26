# -*- coding: utf-8 -*-
"""
Module containing various useful class and functions.

This module contains class for orthogonal basis projection, functions for
error measures and other useful mathematical functions.

Orthogonal basis for projection (see :mod:`pyvi.utilities.orthogonal_basis`)
----------------------------------------------------------------------------
LaguerreBasis :
    Class for Orthogonal Laguerre Basis.
KautzBasis :
    Class for Orthogonal Kautz Basis.
GeneralizedBasis :
    Class for Generalized Orthogonal Basis.
create_orthogonal_basis :
    Returns an orthogonal basis given its poles and its number of elements.
is_valid_basis_instance :
    Checks whether `basis` is a usable instance of a basis.

Error measure functions (see :mod:`pyvi.utilities.measures`)
------------------------------------------------------------
separation_error :
    Returns the relative error between nonlinear orders and their estimates.
identification_error :
    Returns the relative error between kernels and their estimates.
evaluation_error :
    Returns the relative error between a reference signal and an estimation.

Useful mathematic functions (see :mod:`pyvi.utilities.mathbox`)
---------------------------------------------------------------
rms :
    Returns the root-mean-square along given axis.
db :
    Returns the dB value.
safe_db :
    Returns the dB value, with safeguards if numerator or denominator is null.
binomial :
    Binomial coefficient returning an integer.
multinomial :
    Multinomial coefficient returning an integer.
array_symmetrization :
    Symmetrize a multidimensional square array.

Developed for Python 3.6
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

from .mathbox import *
from .measures import *
from .orthogonal_basis import *

__all__ = mathbox.__all__
__all__ += measures.__all__
__all__ += orthogonal_basis.__all__
