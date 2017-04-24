# -*- coding: utf-8 -*-
"""
Tooolbox for useful small math functions.

Notes
-----
@author:    bouvier@ircam.fr
            Damien Bouvier, IRCAM, Paris

Last modified on 24 Apr. 2017
Developed for Python 3.6.1
"""

#==============================================================================
#Importations
#==============================================================================

import numpy as np


#==============================================================================
# Functions
#==============================================================================

def rms(sig):
    """
    Computation of the root-mean-square of a vector.
    """
    return np.sqrt( np.mean(np.abs(sig)**2) )


def db(val, ref=1):
    """
    Conversion to dB.
    """
    return 20 * np.log10(val / ref)

def safe_db(num, den):
    """
    COnversion to dB with verification that neither the denominator nor
    numerator are equal to zero.
    """
    if den == 0:
        return np.Inf
    if num == 0:
        return - np.Inf
    return 20 * np.log10(num / den)