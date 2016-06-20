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


#==============================================================================
# Functions
#==============================================================================
