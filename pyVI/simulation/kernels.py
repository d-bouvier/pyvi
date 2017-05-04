# -*- coding: utf-8 -*-
"""
Description

Set of functions for the numerical simulation of a nonlinear systems given
its state-space representation.

Function
--------
simulation :
    Compute the simulation of a nonlinear system for a given input.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 04 May 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from scipy.fftpack import fftn
from itertools import filterfalse, product
from ..utilities.mathbox import array_symmetrization


#==============================================================================
# Functions
#==============================================================================

def time_kernel_computation(T, fs: int, dimensions: dict, nl_order_max: int,
                            filter_mat, B_m, C_m, D_m, mpq: dict, npq: dict,
                            mpq_combinatoric: dict, npq_combinatoric: dict,
                            holder_bias_mat: dict):
    #TODO docstring

    ####################
    ## Initialization ##
    ####################

    # Input-to-state and input-to-output kernels initialization
    len_kernels = 1 + int(fs * T)
    kernels_in2state = dict()
    kernels_in2out = dict()
    for n in range(1, nl_order_max+1):
        kernels_in2state[n] = np.zeros((dimensions['state'],)+(len_kernels,)*n)

    # Dirac-delta
    dirac = np.zeros((dimensions['input'], len_kernels))
    dirac[0] = 1

    # Enforce good shape when input dimension is 1
    if dimensions['input'] == 1:
        B_m.shape = (dimensions['state'], dimensions['input'])
        D_m.shape = (dimensions['output'], dimensions['input'])
    if dimensions['output'] == 1:
        C_m.shape = (dimensions['output'], dimensions['state'])
        D_m.shape = (dimensions['output'], dimensions['input'])

    ##################################################
    ## Creation of functions for kernel computation ##
    ##################################################

    # Computation of the Mpq/Npq functions (given as tensors)
    def pq_computation(n, p, q, order_set, pq_tensor):
        temp_arg = ()
        min_ind = p + q
        for count in range(p):
            max_ind = min_ind + order_set[count]
            temp_arg += (kernels_in2state[order_set[count]],)
            temp_arg += ([count] + list(range(min_ind, max_ind)),)
            min_ind = max_ind
        for count in range(q):
            temp_arg += (dirac, [p+count, min_ind+count])
        temp_arg += (list(range(p + q + n)),)
        return np.tensordot(pq_tensor, np.einsum(*temp_arg), p+q)

    # Correction of the bias due to ADC converter (with holder of order 0 or 1)
    if len(holder_bias_mat) == 1:
        bias_1sample_lag = holder_bias_mat[0]
        def holder_bias(mpq_output, n):
            idx_in = [slice(None)] + [slice(len_kernels-1)] * n
            idx_out = [slice(None)] + [slice(1, len_kernels)] * n
            kernels_in2state[n][idx_out] += np.tensordot(bias_1sample_lag,
                                                         mpq_output[idx_in], 1)
    elif len(holder_bias_mat) == 2:
        bias_0sample_lag = holder_bias_mat[0] - holder_bias_mat[1]
        bias_1sample_lag = holder_bias_mat[1]
        def holder_bias(mpq_output, n):
            kernels_in2state[n] += np.tensordot(bias_0sample_lag, mpq_output, 1)
            idx_in = [slice(None)] + [slice(len_kernels-1)] * n
            idx_out = [slice(None)] + [slice(1, len_kernels)] * n
            kernels_in2state[n][idx_out] += np.tensordot(bias_1sample_lag,
                                                         mpq_output[idx_in], 1)

    # Filter function
    def filtering(n):
        for ind in range(n,n*(len_kernels-1)+1):
            for indexes in filterfalse(lambda x: sum(x)-ind,
                                       product(range(1, len_kernels),
                                               repeat=n)):
                idx_in = [slice(None)] + [(m-1) for m in indexes]
                idx_out = [slice(None)] + list(indexes)
                kernels_in2state[n][idx_out] += \
                        np.tensordot(filter_mat, kernels_in2state[n][idx_in], 1)

    ########################
    ## Kernel computation ##
    ########################

    ## Dynamical equation ##
    # Linear term
    holder_bias(B_m.dot(dirac), 1)
    filtering(1)
    # Nonlinear terms (due to Mpq functions)
    for n, elt in sorted(mpq_combinatoric.items()):
        for p, q, order_set, nb in elt:
            mpq_output = nb * pq_computation(n, p, q, order_set, mpq[(p, q)])
            holder_bias(mpq_output, n)
        for ind in range(dimensions['state']):
            kernels_in2state[n][ind] = \
                        array_symmetrization(kernels_in2state[n][ind])
        filtering(n)

    ## Output equation ##
    # Linear and nonlinear terms due to matrix C
    for n in range(1, nl_order_max+1):
        kernels_in2out[n] =  np.tensordot(C_m, kernels_in2state[n], 1)
    # Linear term due to matrix D
    kernels_in2out[1] += D_m.dot(dirac)
    # Other nonlinear terms (due to Npq functions)
    for n, elt in sorted(npq_combinatoric.items()):
        for p, q, order_set, nb in elt:
            kernels_in2out[n] += nb * pq_computation(n, p, q, order_set,
                                                     npq[(p, q)])
        for ind in range(dimensions['output']):
            kernels_in2state[n][ind] = \
                        array_symmetrization(kernels_in2state[n][ind])

    ######################
    ## Function outputs ##
    ######################

    if (dimensions['input'] == 1) or (dimensions['output'] == 1):
        for n in range(1, nl_order_max+1):
            kernels_in2state[n] = np.squeeze(kernels_in2state[n])
            kernels_in2out[n] = np.squeeze(kernels_in2out[n])

    return kernels_in2out


def freq_kernel_computation():
    #TODO docstring
    #TODO fonction
    print('Rien')


def freq_kernel_computation_from_time_kernels(volterra_kernels):
    #TODO docstring

    transfer_kernels = dict()
    for n, kernel in volterra_kernels.items():
        transfer_kernels[n] = np.fft.fftshift(fftn(kernel))

    return transfer_kernels