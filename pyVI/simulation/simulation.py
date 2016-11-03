# -*- coding: utf-8 -*-
"""
Summary
-------
Set of functions for the numerical simulation of a nonlinear systems given
its state-space representation.


Multilinear combinatorics
-------------------------
make_list_pq :
    Compute the list of multilinear pq-functions used in each order of
    nonlinearity.
elimination :
    Eliminates the multilinear pq-functions unused in the system.
state_combinatorics :
    Compute, for each Mpq function at a given order n, the different sets of
    state-homogenous-order that are the inputs of the multilinear pq-function.
make_dict_pq_set :
    Return the list of sets characterising multilinear pq-functions used in a
    system.

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


#==============================================================================
# Functions
#==============================================================================

## Auxiliary function for make_dict_pq_set ##
def make_list_pq(nl_order_max):
    """
    Compute the list of multilinear pq-functions used in each order of
    nonlinearity.

    Parameters
    ----------
    nl_order_max : int
        Maximum order of nonlinearity

    Returns
    -------
    list_pq : numpy.ndarray
        Array of shape (N, 3), where N is the number of sets, each of the from
        [n, p, q].
    """

    # Initialisation
    list_pq = np.empty((0, 3), dtype=int)
    # Variable for reporting sets from the previous order
    nb_set_2_report = 0

    # Loop on order of nonlinearity
    for n in range(2, nl_order_max+1):
        # Report previous sets and change the corresponding order
        list_pq = np.concatenate((list_pq, list_pq[-nb_set_2_report-1:-1,:]))
        list_pq[-nb_set_2_report:,0] += 1
        # Loop on all new combination (p,q)
        for q in range(n+1):
            array_tmp = np.array([n, n-q, q])
            array_tmp.shape = (1, 3)
            list_pq = np.concatenate((list_pq, array_tmp))
            # We don't report the use of the pq-function for p = 0
            if not (n == q):
                nb_set_2_report += 1

    return list_pq

def elimination(h_pq_bool, list_pq):
    """
    Eliminates the multilinear pq-functions unused in the system.

    Parameters
    ----------
    h_pq_bool : function (int, int: boolean)
        Indicates, for given p & q, if the multilinear pq-function is used in
        the system.
    list_pq : numpy.ndarray
        Array of all combination [n, p, q].

    Outputs
    -------
    list_pq : numpy.ndarray
        Same array as the ``list_pq`` input array with unused lines deleted.
    """

    # Initialisation
    mask_pq = np.empty(list_pq.shape[0], dtype=bool)
    # Loop on all set combination
    for idx in range(list_pq.shape[0]):
        # In the following:
        # list_pq[idx,0] represents n
        # list_pq[idx,1] represents p
        # list_pq[idx,2] represents q
        mask_pq[idx] = h_pq_bool(list_pq[idx,1], list_pq[idx,2])

    return list_pq[mask_pq]

def state_combinatorics(list_pq, nl_order_max, print_opt=False):
    """
    Compute, for each Mpq function at a given order n, the different sets of
    state-homogenous-order that are the inputs of the multilinear pq-function.
    All sets are created, even those identical in respect to the order (so, if
    the multilinear pq-function are symmetric, there is redudancy).

    Parameters
    ----------
    list_pq : numpy.ndarray
        Array of all combination [n, p, q].
    print_opt : boolean, optional
        Intermediate results printing option.

    Outputs
    -------
    pq_sets : dict
        Dict of all sets [p, q, k] for each order n, where k are the sets
        state-homogenous-order.
    """

    # Initialisation
    pq_sets = {}
    for n in range(2, nl_order_max+1):
        pq_sets[n] = []

    for elt in list_pq:
        # In the following:
        # elt[0] represents n
        # elt[1] represents p
        # elt[2] represents q

        # Maximum value possible for a state order
        k_sum = elt[0] - elt[2]
        # Value needed for the sum of all state order
        k_max = k_sum - elt[1] + 1
        if print_opt: # Optional printing
            print('(n, p, q) = {}'.format(elt))
        # Loop on all possible sets
        for index in np.ndindex( (k_max,)*elt[1] ):
            index = list(map(sum, zip(index, (1,)*elt[1])))
            # Optional printing
            if print_opt:
                print('        Set: {}, Used = {}'.format(index,
                      sum(index) == k_sum))
            # Check if the corresponds to the current (n,p,q) combination
            if sum(index) == k_sum:
                pq_sets[elt[0]].append((elt[1], elt[2], index))

    return pq_sets


def make_dict_pq_set(h_pq_bool, nl_order_max, print_opt=False):
    """
    Return the list of sets characterising multilinear pq-functions used in a
    system.

    Parameters
    ----------
    h_pq_bool : function (int, int: boolean)
        Indicates, for given p & q, if the multilinear pq-function is used in
        the system.
    nl_order_max : int
        Maximum order of nonlinearity.
    print_opt : boolean, optional
        Intermediate results printing option.

    Returns
    -------
    mpq_sets : list
        List of the [n, p, q, k] sets, with:
    - n : int
        Order of nonlinearity where the multilinear pq-function is used.
    - p : int
        Number of state-entry for the multilinear pq-function.
    - q : int
        Number of input-entry for the multilinear pq-function.
    - k : list (of length p)
        Homogenous orders for the state-entries.

    """

    ## Main ##
    list_pq = make_list_pq(nl_order_max)
    list_pq = elimination(h_pq_bool, list_pq)
    pq_sets = state_combinatorics(list_pq, nl_order_max, print_opt)

    # Optional printing
    if print_opt:
        print('')
        for elt in pq_sets:
            print(elt)

    return pq_sets


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
    dict_mpq_set[1] = [(0, 0, [0])]
    if system.mode == 'tensor':
        system.mpq[0, 0] = np.identity(system.dim['state'])
    elif system.mode == 'function':
        system.mpq[0, 0] = lambda a, u: u

    # Compute list of Npq combinations and tensors
    dict_npq_set = make_dict_pq_set(system.is_npq_used, nl_order_max)
    # Add the linear part (respectively the D and C matrices) to the npq dict
    dict_npq_set[1] = [(0, 1, [])]
    if system.mode == 'tensor':
        system.npq[0, 1] = system.D_m
    elif system.mode == 'function':
        system.npq[0, 1] = lambda a, u: np.dot(system.D_m, u)
    for n in range(1, nl_order_max+1):
        dict_npq_set[n].insert(0, (n, 0, [n]))
        if system.mode == 'tensor':
            system.npq[n, 0] = system.C_m
        elif system.mode == 'function':
            system.npq[n, 0] = lambda a, u: np.dot(system.C_m, u)

    ## Dynamical equation - Numerical simulation ##

    # Simulation in tensor mode for ADC converter with holder of order 0
    if (hold_opt == 0) & (system.mode == 'tensor'):
        for k in np.arange(sig_len-1):
            for n, elt in dict_mpq_set.items():
                state_by_order[n,:,k+1] += np.dot(w_filter,
                                                  state_by_order[n,:,k])
                for p, q, order_set in elt:
                    temp_array = system.mpq[(p, q)].copy()
                    for u in range(q):
                        temp_array = np.dot(temp_array, input_sig[:,k])
                    for order in order_set:
                        temp_array = np.dot(temp_array,
                                        state_by_order[order,:,k])
                    state_by_order[n,:,k+1] += np.dot(holder0_bias, temp_array)

    # Simulation in tensor mode for ADC converter with holder of order 1
    elif (hold_opt == 1) & (system.mode == 'tensor'):
        for k in np.arange(sig_len-1):
            for n, elt in dict_mpq_set.items():
                state_by_order[n,:,k+1] += np.dot(w_filter,
                                                  state_by_order[n,:,k])
                for p, q, order_set in elt:
                    temp_array1 = system.mpq[(p, q)].copy()
                    temp_array2 = system.mpq[(p, q)].copy()
                    for u in range(q):
                        temp_array1 = np.dot(temp_array1, input_sig[:,k])
                        temp_array2 = np.dot(temp_array2, input_sig[:,k+1])
                    for order in order_set:
                        temp_array1 = np.dot(temp_array1,
                                            state_by_order[order,:,k])
                        temp_array2 = np.dot(temp_array2,
                                            state_by_order[order,:,k+1])
                    state_by_order[n,:,k+1] += \
                        np.dot(holder1_bias, temp_array1) +\
                        np.dot(holder0_bias - holder1_bias, temp_array2)

    # Simulation in function mode for ADC converter with holder of order 0
    if (hold_opt == 0) & (system.mode == 'function'):
        for n, elt in dict_mpq_set.items():
            for p, q, order_set in elt:
                temp_arg = (sig_len,) + (input_sig,)*q + \
                           tuple(state_by_order[order_set])
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
                temp_arg = (sig_len,) + (input_sig,)*q + \
                           tuple(state_by_order[order_set])
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
                temp_arg = (sig_len,) + (input_sig,)*q + \
                           tuple(state_by_order[order_set])
                output_by_order[n-1,:,:] += system.npq[(p, q)](*temp_arg)

    output_sig = output_by_order.sum(0)

    ## Function outputs ##

    output_sig = output_sig.transpose()
    state_by_order = state_by_order[1:,:,:].transpose(2, 1, 0)
    output_by_order = output_by_order.transpose(2, 1, 0)

    if out == 'output':
        return output_sig
    elif out == 'output_by_order':
        return output_by_order
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

    from pyVI.statespace.statespace import loudspeaker_sica
    from matplotlib import pyplot as plt
    import time

    # Input signal
    fs = 44100
    T = 2
    f1 = 100
    f2 = 300
    amp = 10
    time_vector = np.arange(0, T, step=1/fs)
    f0_vector = np.linspace(f1, f2, num=len(time_vector))
    sig = amp * np.sin(np.pi * f0_vector * time_vector)

    # Simulation
    start1 = time.time()
    out_t = simulation(sig, loudspeaker_sica(mode='tensor'),
                       fs=fs, nl_order_max=3, hold_opt=1)
    end1 = time.time()
    plt.figure('Input- Output (1)')
    plt.subplot(2, 1, 1)
    plt.plot(sig)
    plt.subplot(2, 1, 2)
    plt.plot(out_t)

    start2 = time.time()
    out_f = simulation(sig, loudspeaker_sica(mode='function'),
                       fs=fs, nl_order_max=3, hold_opt=1)
    end2 = time.time()
    plt.figure('Input- Output (2)')
    plt.subplot(2, 1, 1)
    plt.plot(sig)
    plt.subplot(2, 1, 2)
    plt.plot(out_f)

    plt.figure('Difference')
    plt.plot(out_t - out_f)

    print('"tensor" mode: {}s'.format(end1-start1))
    print('"function" mode: {}s'.format(end2-start2))
