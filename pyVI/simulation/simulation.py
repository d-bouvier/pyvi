# -*- coding: utf-8 -*-
"""
Summary
-------
Set of functions for the numerical simulation of a nonlinear systems given
its state-space representation.

System simulation
-----------------
simulation :
    Compute the simulation of a nonlinear system for a given input.

Notes
-----
@author:    bouvier@ircam.fr
            Damien Bouvier, IRCAM, Paris

Last modified on 3 Nov. 2016
Developed for Python 3.5.1
Uses:
 - numpy 1.11.1
 - pivy 0.1
 - scipy 0.18.0
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from scipy import linalg
from pyvi.tools.combinatorics import make_dict_pq_set


#==============================================================================
# Functions
#==============================================================================

def simulation(input_sig, system, fs=44100, nl_order_max=1, hold_opt=1,
               out='output'):
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

    ####################
    ## Initialization ##
    ####################

    # Compute parameters
    sig_len = max(input_sig.shape)
    sampling_time = 1/fs
    w_filter = linalg.expm(system.A_m * sampling_time)
    A_inv = np.linalg.inv(system.A_m)

    dtype = input_sig.dtype
    input_sig = input_sig.copy()

    # Enforce good shape when input dimension is 1
    if system.dim['input'] == 1:
        system.B_m.shape = (system.dim['state'], system.dim['input'])
        system.D_m.shape = (system.dim['output'], system.dim['input'])
        input_sig.shape = (system.dim['input'], sig_len)

    # By-order state and output initialization
    state_by_order = np.zeros((nl_order_max, system.dim['state'], sig_len),
                              dtype)
    output_by_order = np.zeros((nl_order_max, system.dim['output'], sig_len),
                               dtype)

    # Compute list of Mpq combinations and tensors/functions
    dict_mpq_set = make_dict_pq_set(system.is_mpq_used, nl_order_max,
                                    system.sym_bool)
    # Add the linear part (the B matrix) to the mpq dict
    dict_mpq_set[1] = [(0, 1, [], 1)]
    if system.mode == 'tensor':
        system.mpq[0, 1] = system.B_m
    elif system.mode == 'function':
        system.mpq[0, 1] = lambda u: system.B_m.dot(u)

    # Compute list of Npq combinations and tensors/functions
    dict_npq_set = make_dict_pq_set(system.is_npq_used, nl_order_max,
                                    system.sym_bool)
    # Add the linear part (respectively the D and C matrices) to the npq dict
    dict_npq_set[1] = [(0, 1, [], 1)]
    if system.mode == 'tensor':
        system.npq[0, 1] = system.D_m
    elif system.mode == 'function':
        system.npq[0, 1] = lambda u: system.D_m.dot(u)

    for n in range(1, nl_order_max+1):
        dict_npq_set[n].insert(0, (1, 0, [n], 1))
    if system.mode == 'tensor':
        system.npq[1, 0] = system.C_m
    elif system.mode == 'function':
        system.npq[1, 0] = lambda u: system.C_m.dot(u)


    ##########################################
    ## Creation of functions for simulation ##
    ##########################################

    # Computation of the Mpq/Npq functions (given as tensors or functions)
    if system.mode == 'tensor':
        def pq_computation(p, q, order_set, dict_pq):
            temp_arg = ()
            for count in range(p):
                temp_arg += (state_by_order[order_set[count]],)
                temp_arg += ([count, p+q],)
            for count in range(q):
                temp_arg += (input_sig, [p+count, p+q])
            temp_arg += (list(range(p+q+1)),)
            return np.tensordot(dict_pq[(p, q)], np.einsum(*temp_arg), p+q)
    elif system.mode == 'function':
        def pq_computation(p, q, order_set, dict_pq):
            temp_arg = tuple(state_by_order[order_set]) + (input_sig,)*q
            return np.array(dict_pq[(p, q)](*temp_arg))

    # Correction of the bias due to ADC converter (with holder of order 0 or 1)
    if hold_opt == 0:
        bias_1sample_lag = A_inv.dot(w_filter) - A_inv
        def holder_bias(mpq_output):
            return bias_1sample_lag.dot(mpq_output[:,0:-1])
    elif hold_opt == 1:
        A_inv_squared = A_inv.dot(A_inv)
        bias_1sample_lag = A_inv.dot(w_filter) - fs * \
                           (A_inv_squared.dot(w_filter) - A_inv_squared)
        bias_0sample_lag = A_inv.dot(w_filter) - A_inv - bias_1sample_lag
        def holder_bias(mpq_output):
            return bias_1sample_lag.dot(mpq_output[:,0:-1]) + \
                   bias_0sample_lag.dot(mpq_output[:,1::])

    # Filter function (simply a matrix product by 'w_filter')
    def filtering(n):
        for k in np.arange(sig_len-1):
            state_by_order[n-1,:,k+1] += w_filter.dot(state_by_order[n-1,:,k])


    ##########################
    ## Numerical simulation ##
    ##########################

    # Dynamical equation
    for n, elt in dict_mpq_set.items():
        for p, q, order_set, nb in elt:
            mpq_output = nb * \
                    pq_computation(p, q, [m-1 for m in order_set], system.mpq)
            state_by_order[n-1,:,1::] += holder_bias(mpq_output)
        filtering(n)

    # Output equation
    for n, elt in dict_npq_set.items():
        for p, q, order_set, nb in elt:
            output_by_order[n-1,:,:] += nb * \
                    pq_computation(p, q, [m-1 for m in order_set], system.npq)


    ######################
    ## Function outputs ##
    ######################

    # Reshaping state (if necessary)
    if system.dim['state'] == 1:
        state_by_order = state_by_order[:, 0, :]

    # Reshaping output (if necessary)
    if system.dim['output'] == 1:
        output_by_order = output_by_order[:, 0, :]

    # Returns signals chosen by user
    if out == 'output':
        return output_by_order.sum(0)
    elif out == 'output_by_order':
        return output_by_order
    elif out == 'state':
        return state_by_order.sum(0)
    elif out == 'all':
        return output_by_order.sum(0), state_by_order, output_by_order
    else:
        return output_by_order.sum(0)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    from pyvi.simulation.systems import loudspeaker_sica, system_test
    from matplotlib import pyplot as plt
    import time

    ## Test if simulation works correctly ##
    sig_test = np.ones((10000,))
    out = simulation(sig_test, system_test(mode='tensor'),
                     nl_order_max=3, hold_opt=0)
    out = simulation(sig_test, system_test(mode='tensor'),
                     nl_order_max=3, hold_opt=1)
    out = simulation(sig_test, system_test(mode='function'),
                     nl_order_max=3, hold_opt=0)
    out = simulation(sig_test, system_test(mode='function'),
                     nl_order_max=3, hold_opt=1)


    ## Loudspeaker simulation ##
    # Input signal
    fs = 44100
    T = 1
    f1 = 75
    f2 = 125
    amp = 10
    time_vector = np.arange(0, T, step=1/fs)
    f0_vector = np.linspace(f1, f2, num=len(time_vector))
    sig = amp * np.sin(2 * np.pi * f0_vector * time_vector)

    # Simulation
    start1 = time.time()
    out_t = simulation(sig, loudspeaker_sica(mode='tensor', output='current'),
                       fs=fs, nl_order_max=3, hold_opt=0)
    end1 = time.time()
    plt.figure('Input- Output (1)')
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(time_vector, sig)
    plt.subplot(2, 1, 2)
    plt.plot(time_vector, out_t)

    start2 = time.time()
    out_f = simulation(sig, loudspeaker_sica(mode='function', output='current'),
                       fs=fs, nl_order_max=3, hold_opt=0)
    end2 = time.time()
    plt.figure('Input- Output (2)')
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(time_vector, sig)
    plt.subplot(2, 1, 2)
    plt.plot(time_vector, out_f)

    plt.figure('Difference')
    plt.clf()
    plt.plot(time_vector, out_t - out_f)

    print('"tensor" mode: {}s'.format(end1-start1))
    print('"function" mode: {}s'.format(end2-start2))
