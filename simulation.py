# -*- coding: utf-8 -*-
"""
Set of functions for the numerical simulation of a nonlinear systems given
its state-space representation.

@author:    bouvier@ircam.fr
            Damien Bouvier, IRCAM, Paris

Developed for Python 3.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from scipy  import fftpack, linalg
from matplotlib import pyplot as plt


#==============================================================================
# Functions
#==============================================================================

## Auxiliary function for make_list_pq_set ##
def make_list_mpq(nl_order_max):
    """
    Compute the list of Mpq functions used in each order of nonlinearity.
    
    Parameters
    ----------
    nl_order_max : int
        Maximum order of nonlinearity
    
    Returns
    -------
    list_mpq : ndarray
        Array of shape (N, 3), where N is the number of sets, and each set
        is [n, p, q]
    """
    
    # Initialisation
    list_mpq = np.empty((0, 3), dtype=int)
    # Variable for reporting sets from the previous order
    nb_set_2_report = 0
    
    # Loop on order of nonlinearity
    for n in range(2, nl_order_max+1):
        # Report previous sets and change the corresponding order
        list_mpq = np.concatenate((list_mpq, list_mpq[-nb_set_2_report-1:-1,:]))
        list_mpq[-nb_set_2_report:,0] += 1
        # Loop on all new combination (p,q)
        for q in range(n+1):
            array_tmp = np.array([n, n-q, q])
            array_tmp.shape = (1, 3)
            list_mpq = np.concatenate((list_mpq, array_tmp))
            # We don't report the use of the Mpq function for p = 0
            if not (n == q):
                nb_set_2_report += 1

    return list_mpq

def elimination(h_mpq_bool, list_mpq):
    """
    Eliminates the unused Mpq in the system.
    
    Parameters
    ----------
    h_mpq_bool : lambda function
        Returns if the Mpq function is used for a given (p,q)
    list_mpq : ndarray
        Array of all combination [n, p, q]
    
    Outputs
    -------
    list_mpq : ndarray
        Same array as the input array with unused lines deleted
    """

    # Initialisation
    mask_pq = np.empty(list_mpq.shape[0], dtype=bool)
    # Loop on all set combination    
    for idx in range(list_mpq.shape[0]):
        # In the following:
        # list_mpq[idx,0] represents n
        # list_mpq[idx,1] represents p
        # list_mpq[idx,2] represents q
        mask_pq[idx] = h_mpq_bool(list_mpq[idx,1], list_mpq[idx,2])
    
    return list_mpq[mask_pq]

def state_combinatorics(list_mpq, print_opt=False):
    """
    Compute, for each Mpq function at a given order n, the different sets of
    # state-homogenous-order that are the inputs of the Mpq function
    # (all sets are created, even those equals in respect to the order, so, if
    # the Mpq function are symmetric, there is redudancy)
    
    Parameters
    ----------
    list_mpq : ndarray
        Array of all combination [n, p, q]
    
    Outputs
    -------
    mpq_sets : list
        List of sets [n, p, q, k]
    """
    
    # Initialisation
    mpq_sets = []
    for elt in list_mpq:
        # In the following:
        # elt[0] represents n
        # elt[1] represents p
        # elt[2] represents q
    
        # Maximum value possiblee for a state order
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
                elt_bis = list(elt)
#                elt_bis.append(index + [0,] * elt[2])
                elt_bis.append(index)
                mpq_sets.append(elt_bis)
    
    return mpq_sets
    

def make_list_pq_set(h_mpq_bool, nl_order_max, print_opt=False):
    """
    Return the list of sets characterising Mpq functions used in a system.

    Parameters
    ----------
    h_mpq_bool : lambda function
        Function that take two ints (p and q) and returns if the Mpq function
        is used for a given (p,q)
    nl_order_max : int
        Maximum order of nonlinearity
    print_opt : boolean, optional (defaul=False)
        Iintermediate results printing option

    Returns
    -------
    mpq_sets : list
        List of the [n, p, q, k] sets, where
    n : int
        Order of nonlinearity where the Mpq function is used
    p : int
        Number of state-entry for the Mpq multilinear function
    q : int
        Number of input-entry for the Mpq multilinear function
    k : list (of length p)
        Homogenous orders for the state-entry
    """
    
    ## Main ##
    list_mpq = make_list_mpq(nl_order_max)
    list_mpq = elimination(h_mpq_bool, list_mpq)
    mpq_sets = state_combinatorics(list_mpq, print_opt)
    
    # Optional printing
    if print_opt: 
        print('')
        for elt in mpq_sets:
            print(elt)
            
    return mpq_sets


def simulation(input_sig, matrices,
               m_pq=(lambda p,q: False, lambda p,q: None),
               n_pq=(lambda p,q: False, lambda p,q: None),
               sizes=(1, 1, 1), sym_bool=True, fs=44100,
               nl_order_max=1, hold_opt=1):
    """
    Comupte the simulation of a nonlinear system for a given input.
    """
    
    ## Init ##
    # Unpack values
    A_m = matrices[0]
    B_m = matrices[1]
    C_m = matrices[2]
    D_m = matrices[3]

    input_dim = sizes[0]
    state_dim = sizes[1]
    output_dim = sizes[2]

    h_mpq_bool = m_pq[0]
    h_mpq = m_pq[1]
    h_npq_bool = n_pq[0]
    h_npq = n_pq[1]
    
    # Compute parameters
    sig_len = max(input_sig.shape)
    sampling_time = 1/fs
    w_filter = linalg.expm(A_m*sampling_time)
    A_inv = np.linalg.inv(A_m) 
    
    # Enforce good shape when dimension is 1
    if input_dim == 1:
        B_m.shape = (state_dim, input_dim)
        D_m.shape = (output_dim, input_dim)
        input_sig.shape = (input_dim, sig_len)

    # By-order state and output initialization
    state_by_order = np.zeros((nl_order_max+1, state_dim, sig_len))
    output_by_order = np.zeros((nl_order_max, output_dim, sig_len))
    # Put the input signal as order-zero state
    state_by_order[0,:,:] = np.dot(B_m, input_sig)
    
    holder0_bias = np.dot(A_inv, w_filter - np.identity(state_dim))
    if hold_opt == 1:
        holder1_bias = \
                np.dot(A_inv, w_filter) -\
                fs * np.dot(np.dot(A_inv, A_inv),
                            w_filter - np.identity(state_dim))

    # Compute list of Mpq combinations and tensors
    list_mpq_set = make_list_pq_set(h_mpq_bool, nl_order_max)
    dict_mpq = {}
    for idx, elt in enumerate(list_mpq_set):
        if (elt[1], elt[2]) not in dict_mpq:
            dict_mpq[elt[1], elt[2]] = h_mpq(elt[1], elt[2])
    # Add the linear part (the B matrix)
    list_mpq_set.insert(0, [1, 0, 0, [0]])
    dict_mpq[0, 0] = np.identity(state_dim)
    
    # Compute list of Npq combinations and tensors
    list_npq_set = make_list_pq_set(h_npq_bool, nl_order_max)
    dict_npq = {}
    for idx, elt in enumerate(list_npq_set):
        if (elt[1], elt[2]) not in dict_npq:
            dict_npq[elt[1], elt[2]] = h_npq(elt[1], elt[2])

    # Add the linear part (respectively the D and C matrices)
    list_npq_set.insert(0, [1, 0, 1, []])
    dict_npq[0, 1] = D_m
    for n in range(1, nl_order_max+1):
        list_npq_set.insert(0, [n, n, 0, [n]])
        dict_npq[n, 0] = C_m
    

    ## Numerical simulation ##

    # Dynamical equation
    if hold_opt == 0: # Simulation for ADC converter with holder of order 0
        # Main loop (on time indexes)
        for k in np.arange(sig_len-1):        
            for idx, elt in enumerate(list_mpq_set):
                n = elt[0]
                p = elt[1]
                q = elt[2]
                temp_array = dict_mpq[(p, q)].copy()
                for order in range(q):
                    temp_array = np.dot(temp_array, input_sig[:,k])
                for order in elt[3]:
                    temp_array = np.dot(temp_array,
                                        state_by_order[order,:,k])
                state_by_order[n,:,k+1] = \
                        np.dot(w_filter, state_by_order[n,:,k]) +\
                        np.dot(holder0_bias, temp_array)

    elif hold_opt == 1: # Simulation for ADC converter with holder of order 1
        # Main loop (on time indexes)
        for k in np.arange(sig_len-1):        
            for idx, elt in enumerate(list_mpq_set):
                n = elt[0]
                p = elt[1]
                q = elt[2]                       
                temp_array1 = dict_mpq[(p, q)].copy()
                temp_array2 = dict_mpq[(p, q)].copy()
                for order in range(q):
                    temp_array1 = np.dot(temp_array1, input_sig[:,k])
                    temp_array2 = np.dot(temp_array2, input_sig[:,k+1])
                for order in elt[3]:
                    temp_array1 = np.dot(temp_array1,
                                        state_by_order[order,:,k])
                    temp_array2 = np.dot(temp_array2,
                                        state_by_order[order,:,k+1])
                state_by_order[n,:,k+1] = \
                        np.dot(w_filter, state_by_order[n,:,k]) +\
                        np.dot(holder1_bias, temp_array1) +\
                        np.dot(holder0_bias - holder1_bias, temp_array2)
   
    # Output equation
    for k in np.arange(sig_len):
        for idx, elt in enumerate(list_npq_set):
            n = elt[0]
            p = elt[1]
            q = elt[2]
            temp_array = dict_npq[(p, q)].copy()
            for order in range(q):
                temp_array = np.dot(temp_array, input_sig[:,k])
            for order in elt[3]:
                temp_array = np.dot(temp_array,
                                    state_by_order[order,:,k])
                output_by_order[n-1,:,k] += temp_array
    output_sig = output_by_order.sum(0)
    
    # Function outputs
    input_sig.shape = (sig_len, input_dim)
    return output_sig.transpose()

    
def hp_parameters():
    """
    Gives the linear matrices and nonlinear operators of the state-space
    representation for the loudspeaker SICA Z000900
    (http://www.sica.it/media/Z000900C.pdf551d31b7b491e.pdf).
    """ 

    state_dim = 3
    input_dim = 1
    output_dim = 1
    sym_bool = True
    
    ## Linear part ##

    # Electric parameters
    Bl = 2.99 # Electodynamic driving parameter [T.m]
    Re = 5.7 # Electrical resistance of voice coil   [Ohm]
    Le  =   0.11e-3 # Coil inductance [H]
    # Mechanical parameters
    Mms = 1.9e-3; # Mechanical mass [kg]
    Cms = 544e-6; # Mechanical compliance [m.N-1]
    Qms = 4.6;
    k = 1 / Cms # Suspension stiffness [N.m-1]
    Rms = np.sqrt(k * Mms)/Qms; # Mechanical damping and drag force [kg.s-1]   
    # State-space matrices
    A_m = np.array([[-Re/Le, 0, -Bl/Le],
                    [0, 0, 1],
                    [Bl/Mms, -k/Mms, -Rms/Mms]]) # State-to-state matrix
    B_m = np.array([1/Le, 0, 0]); # Input-to-state matrix
    C_m = np.array([[1, 0, 0]]) # State-to-output matrix  
    D_m = np.zeros((output_dim, input_dim)) # Input-to-output matrix    

    ## Nonlinear part ##
    
    # Suspension stiffness polynomial expansion
    k_poly_coeff_v = np.array([k, -554420.0, 989026000])
    mpq_coeff = k_poly_coeff_v/Mms
    # Handles for fonction saying if Mpq and Npq functions are used
    h_mpq_bool = (lambda p, q: q==0) 
    h_npq_bool = (lambda p, q: False)
    # Handles for fonction giving the Mpq tensor
    h_mpq = (lambda p, q: hp_mpq_tensor(p, q, state_dim,  mpq_coeff[p-1]))
    h_npq = (lambda p, q: None)  

    def hp_mpq_tensor(p, q, state_dim, coeff_value):
        """
        Gives the tensor form of the Mpq function (with q = 0).
        """ 
        if q==0:
            mpq_tensor = np.zeros((state_dim,)*(p+1))
            idx = np.concatenate((np.array([2], dtype=int),
                                  np.ones(p, dtype=int)))
            mpq_tensor[tuple(idx)] = coeff_value
            return mpq_tensor
        else:
            return None
            
    return (A_m, B_m, C_m, D_m), (h_mpq_bool, h_mpq), (h_npq_bool, h_npq), \
    (input_dim, state_dim, output_dim), sym_bool


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """
    
    plt.close("all")
    
    # Input signal
    fs = 20000 # Sampling frequency
    T = 0.5 # Duration 
    f1 = 150 # Starting fundamental frequency
    f2 = 210 # Ending fundamental frequency
    A = 10 # Amplitude    
    
    time_vector = np.arange(0, T, step=1/fs)
    f0_vector = np.linspace(f1, f2, num=len(time_vector))
    sig = A * np.cos(2 * np.pi * f0_vector * time_vector)
    
    # Loudspeakers parameters
    matrices, m_pq, n_pq, sizes, sym_bool = hp_parameters()
    
    # Simulation
    output = simulation(sig, matrices, m_pq, n_pq, sizes, sym_bool, fs, 3)

    # Plot    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time_vector, sig, 'b', time_vector, output, 'r')
    plt.subplot(2, 1, 2)
    plt.plot(time_vector, sig, 'b', time_vector, output, 'r')
    plt.xlim([0.1, 0.12])    
    plt.show()
        
    print('Test combinatoire')
    make_list_pq_set(lambda p, q: True, 4, True)
    print('\n\nHP')
    make_list_pq_set(m_pq[0], 4, True)