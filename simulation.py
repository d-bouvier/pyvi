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
from scipy  import fftpack
from matplotlib import pyplot as plt


    
#==============================================================================
# Functions
#==============================================================================

def hp_parameters():
    """
    Gives the linear matrices and nonlinear operators of the state-space
    representation for the loudspeaker SICA Z000900 
    """ 

    state_order = 3
    input_order = 1
    output_order = 1
    boolsym_bool = True
    
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
    D_m = np.zeros((output_order, input_order)) # Input-to-output matrix    

    ## Nonlinear part ##
    
    # Suspension stiffness polynomial expansion
    k_poly_coeff_v = np.array([k, -554420.0, 989026000])
    mpq_coeff = k_poly_coeff_v/Mms
    # Handles for fonction saying if Mpq and Npq functions are used
    h_mpq_bool = (lambda p, q: q==0) 
    h_npq_bool = (lambda p, q: False)
    # Handles for fonction giving the Mpq tensor
    h_mpq = (lambda p, q: hp_mpq_tensor(p, q, state_order,  mpq_coeff(p)))
    h_npq = (lambda p, q: None)

    def hp_mpq_tensor(p, q, state_order, coeff_value):
        """
        Gives the tensor form of the Mpq function (with q = 0)
        """ 
        if q==0:
            Mpq_tensor = np.zeros((state_order,)*(p+1))
            idx = np.concatenate((np.array([2], dtype=int), np.ones(p, dtype=int)))
            Mpq_tensor[tuple(idx)] = coeff_value
            return Mpq_tensor
        else:
            return None
            
    return A_m, B_m, C_m, D_m, h_mpq_bool, h_mpq, h_npq_bool, h_npq


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing
    """
    
    # Input signal
    fs = 44100 # Sampling frequency
    T = 2 # Duration 
    f1 = 220 # Starting fundamental frequency
    f2 = 4400 # Ending fundamental frequency
    A = 1 # Amplitude    
    
    time_vector = np.arange(0, T, step=1/fs)
    f0_vector = np.linspace(f1, f2, num=len(time_vector))
    sig = np.cos(2 * np.pi * f0_vector * time_vector)
    
    plt.figure
    plt.plot(time_vector, sig)
    plt.xlim([0, 0.025])
    plt.show

    # Loudspeakers parameters
    A_m, B_m, C_m, D_m, h_mpq_bool, h_mpq, h_npq_bool, h_npq = hp_parameters()
    print(A_m, B_m, C_m, D_m, h_mpq_bool, h_mpq, h_npq_bool, h_npq, sep='\n')