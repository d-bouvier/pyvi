# -*- coding: utf-8 -*-
"""
Test script for pyvi.utilities.plotbox

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 27 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from pyvi.utilities.plotbox import (plot_sig_io, plot_sig_coll,
                                    plot_kernel_time, plot_kernel_freq)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    print()

    ###################
    ## Data creation ##
    ###################

    vector = np.arange(5, step=1/100)
    sig_1 = np.sin(2 * np.pi * vector)
    sig_2 = np.minimum(sig_1, 0.8)
    sig_3 = np.exp(2j * np.pi * vector)
    sig_4 = np.exp(2j * 1.5 * np.pi * vector)
    M = 41
    time_vec = vector[:M]
    freq_vec = np.fft.fftshift(np.fft.fftfreq(len(time_vec), d=1/100))
    kernel_time_1 = sig_1[:M]
    kernel_time_2 = np.tensordot(kernel_time_1, sig_2[:M], 0)
    kernel_freq_1 = np.fft.fftshift(np.fft.fft(kernel_time_1))
    kernel_freq_2 = np.fft.fftshift(np.fft.fftn(kernel_time_2))


    ############################
    ## Function plot_sig_io() ##
    ############################

    print('Testing plot_sig_io() ...', end=' ')
    plot_sig_io(vector, sig_1, sig_2, title='Test réel', ylim=[-1.1, 1.1])
    plot_sig_io(vector, sig_3, sig_4, title='Test complexe', xlim=[0, 3])
    print('Done.')


    ##########################
    ## Function plot_coll() ##
    ##########################

    print('Testing plot_coll() ...', end=' ')
    plot_sig_coll(vector, np.stack((sig_1, sig_2, sig_1 - sig_2), axis=0),
                  title='Test réel (Collection)', ylim=[-1.1, 1.1],
                  title_plots=['Sinus', 'Cosinus', 'Sinus saturé'])
    plot_sig_coll(vector, np.stack((sig_3, sig_4), axis=0), xlim=[0, 3],
                  title='Test complexe (Collection)')
    print('Done.')


    #################################
    ## Function plot_kernel_time() ##
    #################################

    print('Testing plot_kernel_time() ...', end=' ')
    plot_kernel_time(time_vec, kernel_time_1, title='Test noyau temp - ordre 1')
    plot_kernel_time(time_vec, kernel_time_2, style='contour',
                     title="Test noyau temp - ordre 2 - mode 'contour'")
    plot_kernel_time(time_vec, kernel_time_2, style='surface',
                     title="Test noyau temp - ordre 2 - mode 'surface'")
    plot_kernel_time(time_vec, kernel_time_2, style='wireframe',
                     title="Test noyau temp - ordre 2 - mode 'wireframe'")
    print('Done.')


    #################################
    ## Function plot_kernel_freq() ##
    #################################

    print('Testing plot_kernel_freq() ...', end=' ')
    plot_kernel_freq(freq_vec, kernel_freq_1, title='Test noyau freq - ordre 1')
    plot_kernel_freq(freq_vec, kernel_freq_1, logscale=True,
                     title='Test noyau freq - ordre 1 - logscale')
    plot_kernel_freq(freq_vec, kernel_freq_2, style='contour',
                     title="Test noyau freq - ordre 2 - mode 'contour'")
    plot_kernel_freq(freq_vec, kernel_freq_2, style='surface',
                     title="Test noyau freq - ordre 2 - mode 'surface'")
    plot_kernel_freq(freq_vec, kernel_freq_2, style='wireframe',
                     title="Test noyau freq - ordre 2 - mode 'wireframe'")
    print('Done.')