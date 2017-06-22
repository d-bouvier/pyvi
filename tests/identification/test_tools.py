# -*- coding: utf-8 -*-
"""
Test script for and pyvi.identification.tools

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 22 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from pyvi.identification.tools import error_measure


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    N = 3
    M = 20

    kernels = dict()
    for n in range(1, N+1):
        kernels[n] = np.random.uniform(low=-1.0, high=1.0, size=(M,)*n)

    for sigma in [0, 0.001, 0.01, 0.1, 1]:

        kernels_est = dict()
        for n, h in kernels.items():
            kernels_est[n] = h + np.random.normal(scale=sigma, size=(M,)*n)

        error = error_measure(kernels, kernels_est, db=False)
        error_db = error_measure(kernels, kernels_est)
        print('Added noise factor:', sigma)
        print('Relative error     :', error)
        print('Relative error (dB):', error_db)
        print()