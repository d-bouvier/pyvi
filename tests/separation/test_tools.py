# -*- coding: utf-8 -*-
"""
Test script for pyvi/separation/tools.py

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 27 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from pyvi.separation.tools import error_measure
from mytoolbox.utilities.misc import my_parse_arg_for_tests


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    indent = my_parse_arg_for_tests()


    ##############################
    ## Function error_measure() ##
    ##############################

    print(indent + 'Testing error_measure()...', end=' ')

    N = 3
    L = 1000
    size=(N, L)

    sig = np.random.uniform(low=-1.0, high=1.0, size=size)
    for sigma in [0, 0.001, 0.01, 0.1, 1]:
        sig_est = sig + np.random.normal(scale=sigma, size=size)

        error = error_measure(sig, sig_est, db=False)
        error_db = error_measure(sig, sig_est)

        assert len(error) == size[0], \
            'Error in length of returned error measure.'
        assert len(error_db) == size[0], \
            'Error in length of returned error measure.'
    print('Done.')
