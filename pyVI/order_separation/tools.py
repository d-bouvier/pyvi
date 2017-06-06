# -*- coding: utf-8 -*-
"""
Tools for measuring order separation error.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 6 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from ..utilities.mathbox import rms, safe_db


#==============================================================================
# Functions
#==============================================================================

def estimation_measure(signals_ref, signals_est, db=True):
    """
    Compute the relative error between orders and their estimates.


    """
    error_sig = np.abs(signals_ref - signals_est)
    error_measure = []

    for n in range(signals_est.shape[0]):
        rms_error = rms(error_sig[n])
        rms_ref = rms(signals_ref[n])
        if db:
            val = safe_db(rms_error, rms_ref)
        else:
            val = rms_error / rms_ref
        error_measure.append(val)

    return error_measure