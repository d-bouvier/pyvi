# -*- coding: utf-8 -*-
"""
Set of functions for the numerical simulation of a nonlinear systems given
its state-space representation.

@author:    bouvier@ircam.fr
            Damien Bouvier, IRCAM, Paris

Last modified on 26 Oct. 2016
Developed for Python 3.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from scipy import linalg


#==============================================================================
# Class
#==============================================================================

class System:
    
    def __init__(self, A_m, B_m, C_m, D_m,
                 h_mpq_bool, h_npq_bool, mpq_dict, npq_dict,
                 sym_bool=False, nl_mode='tensor'):
        
        # Initialize the linear part
        self.A_m = A_m        
        self.B_m = B_m
        self.C_m = C_m
        self.D_m = D_m        

        # Extrapolate system dimensions
        self.dim = {'input': B_m.shape[1],
                    'state': A_m.shape[0],
                    'output': C_m.shape[0]}
        
        # Initialize the nonlinear part
        self.is_mpq_used = h_mpq_bool
        self.is_npq_used = h_npq_bool
        self.mpq = mpq_dict
        self.npq = npq_dict
        
        self.sym_bool = sym_bool
        self.nl_mode = nl_mode
        
#==============================================================================
# System parameters
#==============================================================================

""" Loudspeaker SICA Z000900
(http://www.sica.it/media/Z000900C.pdf551d31b7b491e.pdf).
"""

## Physical parameters ##
# Electric parameters
Bl = 2.99 # Electodynamic driving parameter [T.m]
Re = 5.7 # Electrical resistance of voice coil   [Ohm]
Le  =   0.11e-3 # Coil inductance [H]
# Mechanical parameters
Mms = 1.9e-3; # Mechanical mass [kg]
Cms = 544e-6; # Mechanical compliance [m.N-1]
Qms = 4.6;
k = [1/Cms, -554420.0, 989026000] # Suspension stiffness [N.m-1]
Rms = np.sqrt(k[0] * Mms)/Qms; # Mechanical damping and drag force [kg.s-1]   

# State-space matrices
A_m = np.array([[-Re/Le, 0, -Bl/Le],
                [0, 0, 1],
                [Bl/Mms, -k[0]/Mms, -Rms/Mms]]) # State-to-state matrix
B_m = np.array([[1/Le], [0], [0]]); # Input-to-state matrix
C_m = np.array([[1, 0, 0]]) # State-to-output matrix  
D_m = np.zeros((1, 1)) # Input-to-output matrix    

# Handles for fonction saying if Mpq and Npq functions are used
h_mpq_bool = (lambda p, q: (p<=3) & (q==0))
h_npq_bool = (lambda p, q: False)

# Dictionnaries of Mpq & Npq tensors
m20_t = np.zeros((3, 3, 3))
m20_t[2, 1, 1] = k[1]/Mms
m30_t = np.zeros((3, 3, 3, 3))
m30_t[2, 1, 1, 1] = k[2]/Mms
mpq_t_dict = {(2, 0): m20_t, (3, 0): m30_t}
npq_t_dict = dict()

loudspeaker_sica = System(A_m, B_m, C_m, D_m, h_mpq_bool, h_npq_bool,
                          mpq_t_dict, npq_t_dict, sym_bool=True)

# Dictionaries of Mpq & Npq functions
m20_f = lambda x1, x2: np.array([0, 0, k[1]/Mms * x1[1] * x2[1] ])
m30_f = lambda x1, x2, x3: np.array([0, 0, k[2]/Mms * x1[1] * x2[1] * x3[1] ])
mpq_f_dict = {(2, 0): m20_f, (3, 0): m30_f}
npq_f_dict = dict()

loudspeaker_sica_2 = System(A_m, B_m, C_m, D_m, h_mpq_bool, h_npq_bool,
                          mpq_f_dict, npq_f_dict, sym_bool=True,
                          nl_mode='function')


""" Simple system for simulation test. """
test_system = System(np.array([[-1, 0], [1/2, 1/2]]), np.array([[1], [0]]),
                     np.array([[1, 0]]), np.zeros((1, 1)),
                     (lambda p, q: (p+q)<3), (lambda p, q: False),
                     {(2, 0): (lambda x1, x2: np.array([0, x1[0] * x2[0]])),
                      (1, 1): (lambda u, x: np.array([0, u * x[0]])),
                      (0, 2): (lambda u1, u2: np.array([0, u1 * u2]))}, dict(),
                     sym_bool=True, nl_mode='function')

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


def simulation(input_sig, system, fs=44100, nl_order_max=1, hold_opt=1,
               dtype='float', out='output'):
    """
    Compute the simulation of a nonlinear system for a given input.
    """
    
    ## Init ##
    # Compute parameters
    sig_len = max(input_sig.shape)
    sampling_time = 1/fs
    w_filter = linalg.expm(system.A_m * sampling_time)
    A_inv = np.linalg.inv(system.A_m) 
    
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
    list_mpq_set = make_list_pq_set(system.is_mpq_used, nl_order_max)
    # Add the linear part (the B matrix) to the mpq dict
    list_mpq_set.insert(0, [1, 0, 0, [0]])
    system.mpq[0, 0] = np.identity(system.dim['state'])
    
    # Compute list of Npq combinations and tensors
    list_npq_set = make_list_pq_set(system.is_npq_used, nl_order_max)
    # Add the linear part (respectively the D and C matrices) to the npq dict
    list_npq_set.insert(0, [1, 0, 1, []])
    system.npq[0, 1] = system.D_m
    for n in range(1, nl_order_max+1):
        list_npq_set.insert(0, [n, n, 0, [n]])
        system.npq[n, 0] = system.C_m
    
    ## Dynamical equation - Numerical simulation ##

    # Simulation in tensor mode for ADC converter with holder of order 0
    if (hold_opt == 0) & (system.nl_mode == 'tensor'):
        for k in np.arange(sig_len-1):        
            for idx, elt in enumerate(list_mpq_set):
                n = elt[0]
                p = elt[1]
                q = elt[2]
                temp_array = system.mpq[(p, q)].copy()
                for order in range(q):
                    temp_array = np.dot(temp_array, input_sig[:,k])
                for order in elt[3]:
                    temp_array = np.dot(temp_array,
                                        state_by_order[order,:,k])
                state_by_order[n,:,k+1] = \
                        np.dot(w_filter, state_by_order[n,:,k]) +\
                        np.dot(holder0_bias, temp_array)

    # Simulation in tensor mode for ADC converter with holder of order 1
    elif (hold_opt == 1) & (system.nl_mode == 'tensor'):
        for k in np.arange(sig_len-1):        
            for idx, elt in enumerate(list_mpq_set):
                n = elt[0]
                p = elt[1]
                q = elt[2]                       
                temp_array1 = system.mpq[(p, q)].copy()
                temp_array2 = system.mpq[(p, q)].copy()
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
    
    # Simulation in tensor mode for ADC converter with holder of order 0
    if (hold_opt == 0) & (system.nl_mode == 'function'):
        print('Not yet implemented.')

    # Simulation in tensor mode for ADC converter with holder of order 1
    elif (hold_opt == 1) & (system.nl_mode == 'function'):
        print('Not yet implemented.')
   
    ## Dynamical equation - Output simulation ##
    
    for k in np.arange(sig_len):
        for idx, elt in enumerate(list_npq_set):
            n = elt[0]
            p = elt[1]
            q = elt[2]
            temp_array = system.npq[(p, q)].copy()
            for order in range(q):
                temp_array = np.dot(temp_array, input_sig[:,k])
            for order in elt[3]:
                temp_array = np.dot(temp_array,
                                    state_by_order[order,:,k])
            output_by_order[n-1,:,k] += temp_array
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

    # Input signal
    fs = 44100
    T = 2
    f1 = 100
    f2 = 300
    amp = 10   
    time_vector = np.arange(0, T, step=1/fs)
    f0_vector = np.linspace(f1, f2, num=len(time_vector))
    sig = amp * np.cos(np.pi * f0_vector * time_vector)
    
    # Simulation
    out = simulation(sig.copy(), loudspeaker_sica,
                     fs=fs, nl_order_max=3, hold_opt=1)