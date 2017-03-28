# -*- coding: utf-8 -*-
"""
Module for useful small math functions.

Notes
-----
@author:    bouvier@ircam.fr
            Damien Bouvier, IRCAM, Paris

Last modified on 28 Mar. 2016
Developed for Python 3.5.1
Uses:
 - numpy 1.11.1
"""

#==============================================================================
#Importations
#==============================================================================

from numpy as np


#==============================================================================
# Functions
#==============================================================================

def rms(sig):
    """
    Computation of the root-mean-square of a vector.
    """
    return np.sqrt( np.mean(np.abs(sig)**2) )


def safe_db(num, den):
    """
    dB computation with verification that neither the denominator or numerator
    are equal to zero.
    """
    if den == 0:
        return np.Inf
    if num == 0:
        return - np.Inf
    return 20 * np.log10(num / den))