# -*- coding: utf-8 -*-
"""
Test script for and pyvi.identification.tools

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 27 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import string, time
import numpy as np
import itertools as itr
from pyvi.identification.tools import (error_measure, vector_to_kernel,
                                       vector_to_all_kernels,
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


    print('#################################')
    print('## Function vector_to_kernel() ##')
    print('#################################')
    print()

    M = 4

    print('Order 2\n-------')
    n = 2

    h2 = np.arange(1, binomial(M + n - 1, n)+1)
    print('Vector form            :\n', h2)
    print('Triangular kernel form :\n', vector_to_kernel(h2, M, n, form='tri'))
    print('Symmetric kernel form  :\n', vector_to_kernel(h2, M, n))
    print()

    list_ind = list()
    for idx in itr.combinations_with_replacement(string.digits[:M], n):
        list_ind.append(''.join(idx))
    h2s = np.array(list_ind)
    print('Vector form            :\n', h2s)
    print('Triangular kernel form :\n', vector_to_kernel(h2s, M, n, form='tri'))
    print()

    print('Order 3\n-------')
    n = 3

    h3 = np.arange(1, binomial(M + n - 1, n)+1)
    print('Vector form            :\n', h3)
    print('Triangular kernel form :\n', vector_to_kernel(h3, M, n, form='tri'))
    print('Symmetric kernel form  :\n', vector_to_kernel(h3, M, n))
    print()

    list_ind = list()
    for idx in itr.combinations_with_replacement(string.digits[:M], n):
        list_ind.append(''.join(idx))
    h3s = np.array(list_ind)
    print('Vector form            :\n', h3s)
    print('Triangular kernel form :\n', vector_to_kernel(h3s, M, n, form='tri'))
    print()


    print('######################################')
    print('## Function vector_to_all_kernels() ##')
    print('######################################')
    print()

    print('Order 1 to 3\n------------')
    N = 3
    h1s = np.array(list(string.digits[:M]))
    f = np.concatenate((h1s, h2s, h3s), axis=0)
    kernels = vector_to_all_kernels(f, M, N, form='tri')

    print('Vector form            :\n', f)
    print('Triangular kernel form :')
    print(kernels[1])
    print(kernels[2])
    print(kernels[3])

    print('\n')


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