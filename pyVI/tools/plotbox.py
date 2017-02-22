# -*- coding: utf-8 -*-
"""
Toolbox for plots.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 04 Oct. 2016
Developed for Python 3.5.1
Uses:
 - matplolib 1.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import matplotlib.pyplot as plt

#==============================================================================
# Functions
#==============================================================================

def plot_sig_io(input_sig, output_sig, time_vec, name=None,
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


def plot_sig_coll(sig_coll, time_vec, name=None, title_plots=None, 
                  xlim=[None, None], ylim=[None, None], dim=1):
    nb_sig = sig_coll.shape[dim]
    complex_bool = 'complex' in str(sig_coll.dtype)
    if title_plots is None: 
        title_plots = ['Signal {}'.format(n+1) for n in range(nb_sig)]
    
    plt.figure(name)
    plt.clf()
    
    if complex_bool:
        for n in range(nb_sig):
            plt.subplot(nb_sig, 2, 2*n+1)
            plt.plot(time_vec, sig_coll[:, n].real, 'b')
            plt.title(title_plots[n] + ' - real part')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.subplot(nb_sig, 2, 2*n+2)
            plt.plot(time_vec, sig_coll[:, n].imag, 'r')
            plt.title(title_plots[n] + ' - imaginary part')
            plt.xlim(xlim)
            plt.ylim(ylim)
    else:
        for n in range(nb_sig):
            plt.subplot(nb_sig, 1, n+1)
            plt.plot(time_vec, sig_coll[:, n], 'b')
            plt.title(title_plots[n])
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
    sig_2 = np.minimum(sig_1, 0.8)
    sig_3 = np.exp(2j * np.pi * time_vec)
    sig_4 = np.exp(2j * 1.5 * np.pi * time_vec)
    
    plot_sig_io(sig_1, sig_2, time_vec, name='Test réel', ylim=[-1.1, 1.1])
    plot_sig_io(sig_3, sig_4, time_vec, name='Test complexe', xlim=[0, 3])
    
    plot_sig_coll(np.stack((sig_1, sig_2, sig_1 - sig_2), axis=1),
                  time_vec, name='Test réel (Collection)', ylim=[-1.1, 1.1],
                  title_plots=['Sinus', 'Cosinus', 'Sinus saturated'])
    plot_sig_coll(np.stack((sig_3, sig_4), axis=1), time_vec, xlim=[0, 3],
                  name='Test complexe (Collection)')
    