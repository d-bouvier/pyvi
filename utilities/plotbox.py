# -*- coding: utf-8 -*-
"""
Toolbox for plots.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 04 Oct. 2016
Developed for Python 3.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import matplotlib.pyplot as plt

#==============================================================================
# Functions
#==============================================================================

def plot_sig_io(input_sig, output_sig, name, time_vec, 
                xlim=[None, None], ylim=[None, None]):
    complex_bool = 'complex' in str(input_sig.dtype) or \
                   'complex' in str(output_sig.dtype)
    nb_col = 2 if complex_bool else 1
    
    plt.figure(name)
    plt.clf()
    
    plt.subplot(2, nb_col, 1)
    plt.plot(time_vec, input_sig.real, 'b')
    plt.title('Input - Real part' if complex_bool else 'Input')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplot(2, nb_col, 3 if complex_bool else 2)
    plt.plot(time_vec, output_sig.real, 'b')
    plt.title('Output - Real part' if complex_bool else 'Output')
    plt.xlim(xlim)
    plt.ylim(ylim)
    if complex_bool:
        plt.subplot(2, nb_col, 2)
        plt.plot(time_vec, input_sig.imag, 'r')
        plt.title('Input - imaginary part')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.subplot(2, nb_col, 4)
        plt.plot(time_vec, output_sig.imag, 'r')
        plt.title('Output - imaginary part')
        plt.xlim(xlim)
        plt.ylim(ylim)
    plt.show()


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """
    
    import numpy as np
    
    time_vec = np.arange(5, step=1/100)
    sig_1 = np.sin(2 * np.pi * time_vec)
    sig_2 = np.cos(2 * np.pi * time_vec)
    sig_3 = np.exp(2j * np.pi * time_vec)
    sig_4 = np.exp(2j * 1.5 * np.pi * time_vec)
    
    plot_sig_io(sig_1, sig_2, 'Test réel', time_vec, ylim=[-1.1, 1.1])
    plot_sig_io(sig_3, sig_4, 'Test complexe', time_vec, xlim=[0, 3])
    