# -*- coding: utf-8 -*-
"""
Tools for measuring order separation error.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 22 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

from ..utilities.mathbox import rms, safe_db


#==============================================================================
# Functions
#==============================================================================

def error_measure(signals_ref, signals_est, db=True):
    """
    Compute the relative error between orders and their estimates.
    """
    #TODO docstring

    rms_error = rms(signals_ref - signals_est, axis=1)
    rms_ref = rms(signals_ref, axis=1)
    if db:
        return safe_db(rms_error, rms_ref)
    else:
        return rms_error / rms_ref