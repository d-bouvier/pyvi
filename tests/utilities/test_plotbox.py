# -*- coding: utf-8 -*-
"""
Test script for pyvi.utilities.plotbox

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 05 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import os
import argparse
import numpy as np
from matplotlib.pyplot import savefig
from pyvi.utilities.plotbox import (plot_sig_io, plot_sig, plot_coll,
                                    plot_spectrogram, plot_kernel_time,
                                    plot_kernel_freq)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    folder_save_path = os.path.abspath(os.path.dirname(__file__)) + \
                       os.sep + 'test_plotbox'
    if not os.path.isdir(folder_save_path):
        os.mkdir(folder_save_path)
    save_path = folder_save_path + os.sep + '{}.png'


    #####################
    ## Parsing options ##
    #####################

    parser = argparse.ArgumentParser()
    parser.add_argument('-ind', '--indentation', type=int, default=0)
    args = parser.parse_args()
    indent = args.indentation
    ss = ' ' * indent


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

    print(ss + 'Testing plot_sig_io()...', end=' ')
    plot_sig_io(vector, sig_1, sig_2, title='Test plot_sig_io() [réel]',
                ylim=[-1.1, 1.1])
    savefig(save_path.format('plot_sig_io_r'))
    plot_sig_io(vector, sig_3, sig_4, title='Test plot_sig_io() [complexe]',
                xlim=[0, 3])
    savefig(save_path.format('plot_sig_io_c'))
    print('Done.')


    #########################
    ## Function plot_sig() ##
    #########################

    print(ss + 'Testing plot_sig()...', end=' ')
    plot_sig(vector, np.stack((sig_1, sig_2, sig_1 - sig_2), axis=0),
             title='Test plot_sig() [réel]', ylim=[-1.1, 1.1],
             title_plots=['Sinus', 'Sinus saturé', 'Différence'])
    savefig(save_path.format('plot_sig_r'))
    plot_sig(vector, np.stack((sig_3, sig_4), axis=0), xlim=[0, 3],
             title='Test plot_sig() [complexe]')
    savefig(save_path.format('plot_sig_c'))
    print('Done.')


    ##########################
    ## Function plot_coll() ##
    ##########################

    col_1 = np.stack((sig_1, sig_2, sig_1 - sig_2), axis=0)
    col_2 = np.stack((np.imag(sig_3), np.imag(sig_4),
                      np.imag(sig_3 - sig_4)), axis=0)

    print(ss + 'Testing plot_coll()...', end=' ')
    plot_coll(vector, (col_1, col_2), title='Test sig_coll()',
              xtitle=['Colonne 1', 'Colonne 2', 'Colonne 3'],
              ytitle=['Ligne 1', 'Ligne 2', 'Ligne3'])
    savefig(save_path.format('plot_coll'))
    print('Done.')


    #################################
    ## Function plot_kernel_time() ##
    #################################

    print(ss + 'Testing plot_kernel_time()...', end=' ')
    plot_kernel_time(time_vec, kernel_time_1,
                     title='Test plot_kernel_time() [Ordre 1]')
    savefig(save_path.format('plot_kernel_time_1'))
    plot_kernel_time(time_vec, kernel_time_2, style='contour',
                     title="Test plot_kernel_time() [Ordre 2 - 'contour']")
    savefig(save_path.format('plot_kernel_time_2_contour'))
    plot_kernel_time(time_vec, kernel_time_2, style='surface',
                     title="Test plot_kernel_time() [Ordre 2 - 'surface']")
    savefig(save_path.format('plot_kernel_time_2_surface'))
    plot_kernel_time(time_vec, kernel_time_2, style='wireframe',
                     title="Test plot_kernel_time() [Ordre 2 - 'wireframe']")
    savefig(save_path.format('plot_kernel_time_2_wireframe'))
    print('Done.')


    #################################
    ## Function plot_kernel_freq() ##
    #################################

    print(ss + 'Testing plot_kernel_freq()...', end=' ')
    plot_kernel_freq(freq_vec, kernel_freq_1,
                     title='Test plot_kernel_freq() [Ordre 1]')
    savefig(save_path.format('plot_kernel_freq_1'))
    plot_kernel_freq(freq_vec, kernel_freq_1, logscale=True,
                     title='Test plot_kernel_freq() [Ordre 1 - logscale]')
    savefig(save_path.format('plot_kernel_freq_2_logscale'))
    plot_kernel_freq(freq_vec, kernel_freq_2, style='contour',
                     title="Test plot_kernel_freq() [Ordre 2 - 'contour']")
    savefig(save_path.format('plot_kernel_freq_2_contour'))
    plot_kernel_freq(freq_vec, kernel_freq_2, style='surface',
                     title="Test plot_kernel_freq() [Ordre 2 - 'surface']")
    savefig(save_path.format('plot_kernel_freq_2_surface'))
    plot_kernel_freq(freq_vec, kernel_freq_2, style='wireframe',
                     title="Test plot_kernel_freq() [Ordre 2 - 'wireframe']")
    savefig(save_path.format('plot_kernel_freq_2_wireframe'))
    print('Done.')


    #################################
    ## Function plot_spectrogram() ##
    #################################

    fs = 20000
    T = 1
    f1 = 20
    f2 = 10000
    time_vector = np.arange(0, T, step=1/fs)
    k = (f2 -f1)/T
    phi = 2*np.pi * (f1*time_vector + (k/2)*time_vector**2)
    signal = np.sin(phi)

    print(ss + 'Testing plot_spectrogram()...', end=' ')
    opt = {'fs': fs, 'nperseg': 512, 'noverlap': 448, 'nfft': 4096}
    plot_spectrogram(signal, plot_phase=True, unwrap_angle=True,
                     title='Test plot_spectrogram() [unwrapped phase]', **opt)
    savefig(save_path.format('plot_spectrogram_dB'))
    plot_spectrogram(signal, db=False, logscale=True, plot_phase=True,
                     unwrap_angle=False, title='Test plot_spectrogram() ' + \
                     '[unwrapped phase, no dB, logscale]', **opt)
    savefig(save_path.format('plot_spectrogram_logscale'))
    print('Done.')