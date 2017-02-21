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
    system : System
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
        - ``output_sig``, ``state_by_order``, and ``output_by_order`` (if \
        ``out`` == 'all')

    """

    ## Init ##
    # Compute parameters
    sig_len = max(input_sig.shape)
    sampling_time = 1/fs
    w_filter = linalg.expm(system.A_m * sampling_time)
    A_inv = np.linalg.inv(system.A_m)

    dtype = input_sig.dtype
    input_sig = input_sig.copy()

    # Enforce good shape when dimension is 1
    if system.dim['input'] == 1:
        system.B_m.shape = (system.dim['state'], system.dim['input'])
        system.D_m.shape = (system.dim['output'], system.dim['input'])
        input_sig.shape = (system.dim['input'], sig_len)

    # By-order state and output initialization
    state_by_order = np.zeros((nl_order_max+1, system.dim['state'], sig_len),
                              dtype)
    output_by_order = np.zeros((nl_order_max, system.dim['output'], sig_len),
                               dtype)
    # Put the input signal as order-zero state
    state_by_order[0,:,:] = np.dot(system.B_m, input_sig)

    holder0_bias = np.dot(A_inv, w_filter - np.identity(system.dim['state']))
    if hold_opt == 1:
        holder1_bias = \
                np.dot(A_inv, w_filter) -\
                fs * np.dot(np.dot(A_inv, A_inv),
                            w_filter - np.identity(system.dim['state']))

    # Compute list of Mpq combinations and tensors
    dict_mpq_set = make_dict_pq_set(system.is_mpq_used, nl_order_max)
    # Add the linear part (the B matrix) to the mpq dict
    dict_mpq_set[1] = [(1, 0, [0])]
    if system.mode == 'tensor':
        system.mpq[1, 0] = np.identity(system.dim['state'])
    elif system.mode == 'function':
        system.mpq[1, 0] = lambda u: u

    # Compute list of Npq combinations and tensors
    dict_npq_set = make_dict_pq_set(system.is_npq_used, nl_order_max)
    # Add the linear part (respectively the D and C matrices) to the npq dict
    dict_npq_set[1] = [(0, 1, [])]
    if system.mode == 'tensor':
        system.npq[0, 1] = system.D_m
    elif system.mode == 'function':
        system.npq[0, 1] = lambda u: np.dot(system.D_m, u)
    for n in range(1, nl_order_max+1):
        dict_npq_set[n].insert(0, (n, 0, [n]))
        if system.mode == 'tensor':
            system.npq[n, 0] = system.C_m
        elif system.mode == 'function':
            system.npq[n, 0] = lambda u: np.dot(system.C_m, u)

    ## Dynamical equation - Numerical simulation ##

    # Simulation in tensor mode for ADC converter with holder of order 0
    if (hold_opt == 0) & (system.mode == 'tensor'):
        for n, elt in dict_mpq_set.items():
            for p, q, order_set in elt:
                temp_arg = ()
                for count in range(p):
                    temp_arg += (state_by_order[order_set[count]],)
                    temp_arg += ([count, int(p+q)],)
                for count in range(q):
                    temp_arg += (input_sig, [p+count, int(p+q)])
                temp_arg += (list(range(p+q+1)),)
                temp_array = np.tensordot(system.mpq[(p, q)],
                                          np.einsum(*temp_arg), p+q)
                state_by_order[n,:,1::] += \
                        np.dot(holder0_bias, temp_array)[:,0:-1]
            for k in np.arange(sig_len-1):
                state_by_order[n,:,k+1] += np.dot(w_filter,
                                                  state_by_order[n,:,k])

    # Simulation in tensor mode for ADC converter with holder of order 1
    elif (hold_opt == 1) & (system.mode == 'tensor'):
        for n, elt in dict_mpq_set.items():
            for p, q, order_set in elt:
                temp_arg = ()
                for count in range(p):
                    temp_arg += (state_by_order[order_set[count]],)
                    temp_arg += ([count, int(p+q)],)
                for count in range(q):
                    temp_arg += (input_sig, [p+count, int(p+q)])
                temp_arg += (list(range(p+q+1)),)
                temp_array = np.tensordot(system.mpq[(p, q)],
                                          np.einsum(*temp_arg), p+q)
                state_by_order[n,:,1::] += \
                        np.dot(holder1_bias, temp_array)[:,0:-1] +\
                        np.dot(holder0_bias - holder1_bias, temp_array)[:,1::]
            for k in np.arange(sig_len-1):
                state_by_order[n,:,k+1] += np.dot(w_filter,
                                                  state_by_order[n,:,k])

    # Simulation in function mode for ADC converter with holder of order 0
    if (hold_opt == 0) & (system.mode == 'function'):
        for n, elt in dict_mpq_set.items():
            for p, q, order_set in elt:
                temp_arg = (input_sig,)*q + tuple(state_by_order[order_set])
                temp_array = system.mpq[(p, q)](*temp_arg)
                state_by_order[n,:,1::] += \
                        np.dot(holder0_bias, temp_array)[:,0:-1]
            for k in np.arange(sig_len-1):
                state_by_order[n,:,k+1] += np.dot(w_filter,
                                                  state_by_order[n,:,k])

    # Simulation in function mode for ADC converter with holder of order 1
    elif (hold_opt == 1) & (system.mode == 'function'):
        for n, elt in dict_mpq_set.items():
            for p, q, order_set in elt:
                temp_arg = (input_sig,)*q + tuple(state_by_order[order_set])
                temp_array = system.mpq[(p, q)](*temp_arg)
                state_by_order[n,:,1::] += \
                        np.dot(holder1_bias, temp_array)[:,0:-1] +\
                        np.dot(holder0_bias - holder1_bias, temp_array)[:,1::]
            for k in np.arange(sig_len-1):
                state_by_order[n,:,k+1] += np.dot(w_filter,
                                                  state_by_order[n,:,k])

    ## Output equation - Numerical simulation ##

    if system.mode == 'tensor':
        for k in np.arange(sig_len):
            for n, elt in dict_npq_set.items():
                for p, q, order_set in elt:
                    temp_array = system.npq[(p, q)].copy()
                    for u in range(q):
                        temp_array = np.dot(temp_array, input_sig[:,k])
                    for order in order_set:
                        temp_array = np.dot(temp_array,
                                            state_by_order[order,:,k])
                    output_by_order[n-1,:,k] += temp_array

    elif system.mode == 'function':
        for n, elt in dict_npq_set.items():
            for p, q, order_set in elt:
                temp_arg = (input_sig,)*q + tuple(state_by_order[order_set])
                output_by_order[n-1,:,:] += system.npq[(p, q)](*temp_arg)

    ## Function outputs ##

    if system.dim['output'] == 1:
        output_by_order = output_by_order[:, 0, :]
    output_sig = output_by_order.sum(0)

    if system.dim['state'] == 1:
        state_by_order = state_by_order[1:,0,:]
    else:
        state_by_order = state_by_order[1:,:,:]
    state_sig = state_by_order.sum(0)

    if out == 'output':
        return output_sig
    elif out == 'output_by_order':
        return output_by_order
    if out == 'state':
        return state_sig
    elif out == 'all':
        return output_sig, state_by_order, output_by_order
    else:
        return output_sig

#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    from pyvi.systems import loudspeaker_sica
    from matplotlib import pyplot as plt
    import time

    # Input signal
    fs = 44100
    T = 1
    f1 = 100
    f2 = 300
    amp = 10
    time_vector = np.arange(0, T, step=1/fs)
    f0_vector = np.linspace(f1, f2, num=len(time_vector))
    sig = amp * np.sin(np.pi * f0_vector * time_vector)

    # Simulation
    start1 = time.time()
    out_t = simulation(sig, loudspeaker_sica(mode='tensor'),
                       fs=fs, nl_order_max=3, hold_opt=0)
    end1 = time.time()
    plt.figure('Input- Output (1)')
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(sig)
    plt.subplot(2, 1, 2)
    plt.plot(out_t)

    start2 = time.time()
    out_f = simulation(sig, loudspeaker_sica(mode='function'),
                       fs=fs, nl_order_max=3, hold_opt=0)
    end2 = time.time()
    plt.figure('Input- Output (2)')
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(sig)
    plt.subplot(2, 1, 2)
    plt.plot(out_f)

    plt.figure('Difference')
    plt.clf()
    plt.plot(out_t - out_f)

    print('"tensor" mode: {}s'.format(end1-start1))
    print('"function" mode: {}s'.format(end2-start2))
