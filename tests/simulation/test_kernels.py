# -*- coding: utf-8 -*-
"""
Test script for and pyvi.simulation.kernels

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 2 May. 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from pyvi.system.dict import test, nl_damping
from pyvi.utilities.plotbox import plot_kernel_time


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    # Parameters
    options = {'fs': 700,
               'nl_order_max': 3,
               'holder_order': 1}
    T =0.05
    time_vec = np.arange(0, T + (1/options['fs']), step=1/options['fs'])

    # Test system
    system_test = test(mode='numeric')
    t_kernels = system_test.compute_kernels(T, which='time', **options)

    # Second-order system with nonlinear damping
    system = nl_damping(gain=1, f0=100, damping=0.2, nl_coeff=[1e-1, 3e-5])
    time_kernels = system.compute_kernels(T, which='time', **options)

    plot_kernel_time(time_vec, time_kernels[1])
    plot_kernel_time(time_vec, time_kernels[2], style='surface')
