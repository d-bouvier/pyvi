# -*- coding: utf-8 -*-
"""
Module forthe numerical simulation of a systems given by a NumericalStateSpace.

Function
--------
simulation :
    Compute the simulation of a nonlinear system for a given input.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 06 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np


#==============================================================================
# Functions
#==============================================================================

def simulation(input_sig, dimensions: dict, nl_order_max: int,
               filter_mat, B_m, C_m, D_m, mpq: dict, npq: dict,
               mpq_combinatoric: dict, npq_combinatoric: dict,
               holder_bias_mat: dict, out_opt: str):
    """
    Compute the simulation of a nonlinear system for a given input.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Input signal.
    system : StateSpace
        Parameters of the system to simulate.
    fs : int, optional
        Sampling frequency.
    nl_order_max : int, optional
        Maximum order of nonlinearity to take into account.
    hold_opt : {0, 1}, optional
        Type of sample-holder of the ADC converter to emulate.
    out : {'output', 'output_by_order', 'all'}, optional
        Option to choose the output.

    Returns
    -------
    output_sig : numpy.ndarray
        Output of the system.
    output_by_order : numpy.ndarray
        Output of the system, separated in each order of nonlinearity.
    state_by_order : numpy.ndarray
        States of the system, separated in each order of nonlinearity.

    In function of the ``out`` option, this function returns:
        - ``output_sig`` (if ``out`` == 'output')
        - ``output_by_order`` (if ``out`` == 'output_by_order')
        - ``state`` (if ``out`` == 'state')
        - ``output_sig``, ``state_by_order``, and ``output_by_order`` (if \
        ``out`` == 'all')

    """
    #TODO update docstring

    ####################
    ## Initialization ##
    ####################

    input_sig = input_sig.copy()
    dtype = input_sig.dtype

    # Enforce good shape when input dimension is 1
    if dimensions['input'] == 1:
        sig_len = input_sig.shape[0]
        B_m.shape = (dimensions['state'], dimensions['input'])
        D_m.shape = (dimensions['output'], dimensions['input'])
        input_sig.shape = (dimensions['input'], sig_len)
    else:
        sig_len = input_sig.shape[0]

    # By-order state and output initialization
    state_by_order = np.zeros((nl_order_max, dimensions['state'], sig_len),
                              dtype)
    output_by_order = np.zeros((nl_order_max, dimensions['output'], sig_len),
                               dtype)

    ##########################################
    ## Creation of functions for simulation ##
    ##########################################

    # Computation of the Mpq/Npq functions (given as tensors)
    def pq_computation(p, q, order_set, pq_tensor):
        temp_arg = ()
        for count in range(p):
            temp_arg += (state_by_order[order_set[count]],)
            temp_arg += ([count, p+q],)
        for count in range(q):
            temp_arg += (input_sig, [p+count, p+q])
        temp_arg += (list(range(p+q+1)),)
        return np.tensordot(pq_tensor, np.einsum(*temp_arg), p+q)

    # Correction of the bias due to ADC converter (with holder of order 0 or 1)
    if len(holder_bias_mat) == 1:
        bias_1sample_lag = holder_bias_mat[0]
        def holder_bias(mpq_output, n):
            state_by_order[n-1,:,1::] += np.dot(bias_1sample_lag,
                                                mpq_output[:,0:-1])
    elif len(holder_bias_mat) == 2:
        bias_0sample_lag = holder_bias_mat[0] - holder_bias_mat[1]
        bias_1sample_lag = holder_bias_mat[1]
        def holder_bias(mpq_output, n):
            state_by_order[n-1,:,:] += np.dot(bias_0sample_lag, mpq_output)
            state_by_order[n-1,:,1::] += np.dot(bias_1sample_lag,
                                                mpq_output[:,0:-1])

    # Filter function (simply a matrix product by 'filter_mat')
    def filtering(n):
        for k in np.arange(sig_len-1):
            state_by_order[n-1,:,k+1] += filter_mat.dot(state_by_order[n-1,:,k])

    ##########################
    ## Numerical simulation ##
    ##########################

    ## Dynamical equation ##
    # Linear state
    holder_bias(np.dot(B_m, input_sig), 1)
    filtering(1)
    # Nonlinear states (due to Mpq functions)
    for n, elt in sorted(mpq_combinatoric.items()):
        for p, q, order_set, nb in elt:
            mpq_output = nb * \
                    pq_computation(p, q, [m-1 for m in order_set], mpq[(p, q)])
            holder_bias(mpq_output, n)
        filtering(n)

    ## Output equation ##
    # Output term due to matrix D
    output_by_order[0] += np.dot(D_m, input_sig)
    # Output terms due to matrix C
    for n in range(nl_order_max):
        output_by_order[n] += np.dot(C_m, state_by_order[n])
    # Other nonlinear output terms (due to Npq functions)
    for n, elt in sorted(npq_combinatoric.items()):
        for p, q, order_set, nb in elt:
            output_by_order[n-1,:,:] += nb * \
                    pq_computation(p, q, [m-1 for m in order_set], npq[(p, q)])

    ######################
    ## Function outputs ##
    ######################

    # Reshaping state (if necessary)
    if dimensions['state'] == 1:
        state_by_order = state_by_order[:, 0, :]

    # Reshaping output (if necessary)
    if dimensions['output'] == 1:
        output_by_order = output_by_order[:, 0, :]

    if out_opt == 'output':
        return output_by_order.sum(0)
    if out_opt == 'output_by_order':
        return output_by_order
    if out_opt == 'state':
        return state_by_order.sum(0)
    if out_opt == 'state_by_order':
        return state_by_order
    if out_opt == 'all':
        return output_by_order, state_by_order