# -*- coding: utf-8 -*-
"""
Test script for pyvi/identification/tools.py

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 21 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import string
import time
import argparse
import itertools as itr
import numpy as np
from pyvi.identification.tools import (error_measure, nb_coeff_in_kernel,
                                       nb_coeff_in_all_kernels,
                                       vector_to_kernel,
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

    #####################
    ## Parsing options ##
    #####################

    parser = argparse.ArgumentParser()
    parser.add_argument('-ind', '--indentation', type=int, default=0)
    args = parser.parse_args()
    indent = args.indentation
    ss = ' ' * indent


    ##############################
    ## Function error_measure() ##
    ##############################

    print(ss + 'Testing error_measure()...', end=' ')
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
        assert len(error) == N, 'Error in length of returned error measure.'
        assert len(error_db) == N, 'Error in length of returned error measure.'
    print('Done.')


    ###################################
    ## Function nb_coeff_in_kernel() ##
    ###################################

    N = 5
    M = 20

    print(ss + 'Testing nb_coeff_in_kernel()...', end=' ')
    for n in range(1, N+1):
        for M in range(1, M+1):
            nb_coeff_1 = binomial(M + n - 1, n)
            nb_coeff_2 = M**n
            nb_coeff_sym = nb_coeff_in_kernel(M, n, form='sym')
            nb_coeff_tri = nb_coeff_in_kernel(M, n, form='tri')
            nb_coeff_raw = nb_coeff_in_kernel(M, n, form=None)
            assert nb_coeff_sym == nb_coeff_1, \
                'Returns wrong number of coefficient for symmetric form.'
            assert nb_coeff_tri == nb_coeff_1, \
                'Returns wrong number of coefficient for triangular form.'
            assert nb_coeff_raw == nb_coeff_2, \
                'Returns wrong number of coefficient for raw form.'
    print('Done.')


    ########################################
    ## Function nb_coeff_in_all_kernels() ##
    ########################################

    Nmax = 5

    print(ss + 'Testing nb_coeff_in_all_kernels()...', end=' ')
    for N in range(1, Nmax+1):
        for M in range(1, M+1):
            nb_coeff_1 = binomial(M + N, N) - 1
            nb_coeff_2 = sum([M**n for n in range(1, N+1)])
            nb_coeff_sym = nb_coeff_in_all_kernels(M, N, form='sym')
            nb_coeff_tri = nb_coeff_in_all_kernels(M, N, form='tri')
            nb_coeff_raw = nb_coeff_in_all_kernels(M, N, form=None)
            assert nb_coeff_sym == nb_coeff_1, \
                'Returns wrong number of coefficient for symmetric form.'
            assert nb_coeff_tri == nb_coeff_1, \
                'Returns wrong number of coefficient for triangular form.'
            assert nb_coeff_raw == nb_coeff_2, \
                'Returns wrong number of coefficient for raw form.'
    print('Done.')


    #################################
    ## Function vector_to_kernel() ##
    #################################

    print(ss + 'Testing vector_to_kernel()...', end=' ')
    M = 4

    # Order 2
    h2 = np.arange(1, binomial(M + 1, 2)+1)
    h2tri = np.array([[1, 2, 3, 4],
                      [0, 5, 6, 7],
                      [0, 0, 8, 9],
                      [0, 0, 0, 10]])
    h2sym = np.array([[1, 1, 1.5, 2],
                      [1, 5, 3, 3.5],
                      [1.5, 3, 8, 4.5],
                      [2, 3.5, 4.5, 10]])
    h3 = np.arange(1, binomial(M + 2, 3)+1)
    h3tri = np.array([[[1,  2,  3,  4],
                       [0,  5,  6,  7],
                       [0,  0,  8,  9],
                       [0,  0,  0, 10]],
                      [[0,  0,  0,  0],
                       [0, 11, 12, 13],
                       [0,  0, 14, 15],
                       [0,  0,  0, 16]],
                      [[0,  0,  0,  0],
                       [0,  0,  0,  0],
                       [0,  0, 17, 18],
                       [0,  0,  0, 19]],
                      [[0,  0,  0,  0],
                       [0,  0,  0,  0],
                       [0,  0,  0,  0],
                       [0,  0,  0, 20]]])
    h3sym = np.array([[[1., 2/3, 1, 4/3],
                       [2/3, 5/3, 1, 7/6],
                       [1, 1, 8/3, 1.5],
                       [4/3, 7/6, 1.5, 10/3]],
                      [[2/3, 5/3, 1, 7/6],
                       [5/3, 11, 4, 13/3],
                       [1, 4, 14/3, 2.5],
                       [7/6, 13/3, 2.5, 16/3]],
                      [[1, 1, 8/3, 1.5],
                       [1, 4, 14/3, 2.5],
                       [8/3, 14/3, 17, 6],
                       [1.5, 2.5, 6, 19/3]],
                      [[4/3, 7/6, 1.5, 10/3],
                       [7/6, 13/3, 2.5, 16/3],
                       [1.5, 2.5, 6, 19/3],
                       [10/3, 16/3, 19/3, 20]]])
    list_ind_2 = list()
    list_ind_3 = list()
    for idx in itr.combinations_with_replacement(string.digits[:M], 2):
        list_ind_2.append(''.join(idx))
    for idx in itr.combinations_with_replacement(string.digits[:M], 3):
        list_ind_3.append(''.join(idx))
    h2s = np.array(list_ind_2)
    h3s = np.array(list_ind_3)
    h2s_tri = np.array([['00', '01', '02', '03'],
                        ['', '11', '12', '13'],
                        ['', '', '22', '23'],
                        ['', '', '', '33']])
    h3s_tri = np.array([[['000', '001', '002' ,'003'],
                         ['', '011', '012', '013'],
                         ['', '', '022', '023'],
                         ['', '', '', '033']],
                        [['', '', '', ''],
                         ['', '111', '112', '113'],
                         ['', '', '122', '123'],
                         ['', '', '', '133']],
                        [['', '', '', ''],
                         ['', '', '', ''],
                         ['', '', '222', '223'],
                         ['', '', '', '233']],
                        [['', '', '', ''],
                         ['', '', '', ''],
                         ['', '', '', ''],
                         ['', '', '', '333']]])
    assert np.all(vector_to_kernel(h2, M, 2, form='tri') == h2tri), \
        'Error of computation in kernel under its triangular form.'
    assert np.all(vector_to_kernel(h2, M, 2, form='sym') == h2sym), \
        'Error of computation in kernel under its symmetric form.'
    assert np.all(vector_to_kernel(h3, M, 3, form='tri') == h3tri), \
        'Error of computation in kernel under its triangular form.'
    assert np.all(vector_to_kernel(h3, M, 3, form='sym') == h3sym), \
        'Error of computation in kernel under its symmetric form.'
    assert np.all(vector_to_kernel(h2s, M, 2, form='tri') == h2s_tri), \
        'Error of computation in kernel under its triangular form.'
    assert np.all(vector_to_kernel(h3s, M, 3, form='tri') == h3s_tri), \
        'Error of computation in kernel under its triangular form.'
    print('Done.')


    ######################################
    ## Function vector_to_all_kernels() ##
    ######################################

    print(ss + 'Testing vector_to_all_kernels()...', end=' ')
    N = 3
    h1s = np.array(list(string.digits[:M]))
    f = np.concatenate((h1s, h2s, h3s), axis=0)
    kernels = vector_to_all_kernels(f, M, N, form='tri')
    h1s_tri = np.array(['0', '1', '2', '3'])
    assert np.all(kernels[1] == h1s_tri), \
        'Error of computation in vector_to_all_kernels().'
    assert np.all(kernels[2] == h2s_tri), \
        'Error of computation in vector_to_all_kernels().'
    assert np.all(kernels[3] == h3s_tri), \
        'Error of computation in vector_to_all_kernels().'
    print('Done.')


    #####################################
    ## Functions volterra_basis_by_*() ##
    #####################################

    length = 100
    N = 4
    M = 25
    sig = np.arange(length)
    sig_cplx = np.arange(length) + 2j * np.arange(length)

    print(ss + 'Testing volterra_basis_by_order()...', end=' ')
    order_r = volterra_basis_by_order(sig, M, N)
    order_c = volterra_basis_by_order(sig_cplx, M, N)
    print('Done.')

    print(ss + 'Testing volterra_basis_by_term()...', end=' ')
    term_r = volterra_basis_by_term(sig, M, N)
    term_c = volterra_basis_by_term(sig_cplx, M, N)
    for n in range(1, N+1):
        assert np.all(order_r[n] == term_r[(n, 0)]), \
            "Divergence of results for 'order' and term' methods."
        assert np.all(order_c[n] == term_c[(n, 0)]), \
            "Divergence of results for 'order' and term' methods."
        for q in range(0, 1+n//2):
            assert np.all(term_r[(n, 0)] == term_r[(n, q)]), \
                "Divergence of results for 'term' method on real signal."
    print('Done.')
