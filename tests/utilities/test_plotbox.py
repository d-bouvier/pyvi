# -*- coding: utf-8 -*-
"""
Test script for pyvi.utilities.plotbox

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 27 Apr. 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from pyvi.utilities.plotbox import plot_sig_io, plot_sig_coll, plot_kernel_time


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """


    time_vec = np.arange(5, step=1/100)
    sig_1 = np.sin(2 * np.pi * time_vec)
    sig_2 = np.minimum(sig_1, 0.8)
    sig_3 = np.exp(2j * np.pi * time_vec)
    sig_4 = np.exp(2j * 1.5 * np.pi * time_vec)

    plot_sig_io(sig_1, sig_2, time_vec, name='Test réel', ylim=[-1.1, 1.1])
    plot_sig_io(sig_3, sig_4, time_vec, name='Test complexe', xlim=[0, 3])

    plot_sig_coll(np.stack((sig_1, sig_2, sig_1 - sig_2), axis=1),
                  time_vec, name='Test réel (Collection)', ylim=[-1.1, 1.1],
                  title_plots=['Sinus', 'Cosinus', 'Sinus saturé'])
    plot_sig_coll(np.stack((sig_3, sig_4), axis=1), time_vec, xlim=[0, 3],
                  name='Test complexe (Collection)')

    M = 40
    vec = time_vec[:M]
    sig_5 = sig_1[:M]
    sig_6 = sig_2[:M]

    plot_kernel_time(vec, sig_5, title='Test noyau temp - ordre 1')
    plot_kernel_time(vec, np.tensordot(sig_5, sig_6, 0), style='contour',
                     title="Test noyau temp - ordre 2 - mode 'contour'")
    plot_kernel_time(vec, np.tensordot(sig_5, sig_6, 0), style='surface',
                     title="Test noyau temp - ordre 2 - mode 'surface'")
    plot_kernel_time(vec, np.tensordot(sig_5, sig_6, 0), style='wireframe',
                     title="Test noyau temp - ordre 2 - mode 'wireframe'")


