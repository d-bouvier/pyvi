# -*- coding: utf-8 -*-
"""
Test script for and pyvi.identification.tools

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 23 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import string, time
import numpy as np
import itertools as itr
from pyvi.identification.tools import (error_measure, vector_to_kernel,
                                       volterra_basis_by_order,
                                       volterra_basis_by_term)
from pyvi.utilities.mathbox import binomial


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    print('##############################')
    print('## Function error_measure() ##')
    print('##############################')
    print()

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
        print('Added noise factor :', sigma)
        print('Relative error     :', error)
        print('Relative error (dB):', error_db)
        print()


    print('###################################')
    print('## Functions volterra_basis***() ##')
    print('###################################')
    print()

    length = 100
    sig = np.arange(length)
    sig_cplx = np.arange(length) + 2j * np.arange(length)

    for N in range(3, 5):
        for M in [10, 15, 20, 25]:
            print('Order max {}, Memory length {}:'.format(N, M))
            print('------------------------------')

            deb2 = time.time()
            volterra_basis_by_order(sig, M, N)
            fin2 = time.time()
            print('Function volterra_basis_by_order() :', fin2 - deb2)

            deb3 = time.time()
            volterra_basis_by_term(sig_cplx, M, N)
            fin3 = time.time()
            print('Function volterra_basis_by_term()  :', fin3 - deb3)

            print()