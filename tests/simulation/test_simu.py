# -*- coding: utf-8 -*-
"""
Test script for and pyvi.simulation.simu

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 07 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
import time
from pyvi.system.dict import test, loudspeaker_sica, nl_damping
from pyvi.simulation.simu import SimulationObject as SimuObj
from pyvi.utilities.plotbox import (plot_sig_io, plot_sig_coll,
                                    plot_kernel_time, plot_kernel_freq)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    ########################################
    ## Test if simulation works correctly ##
    ########################################

    sig = np.ones((10000,))
    system = test(mode='numeric')

    out1 = SimuObj(system, holder_order=0, resampling=False).simulation(sig)
    out2 = SimuObj(system, holder_order=1, resampling=False).simulation(sig)
    out3 = SimuObj(system, holder_order=0, resampling=True).simulation(sig)
    out4 = SimuObj(system, holder_order=1, resampling=True).simulation(sig)

    assert out1.shape == sig.shape, 'Shape error in simulation output' + \
            ' with holder of order 0 and without resampling.'
    assert out2.shape == sig.shape, 'Shape error in simulation output' + \
            ' with holder of order 1 and without resampling.'
    assert out3.shape == sig.shape, 'Shape error in simulation output' + \
            ' with holder of order 0 and with resampling.'
    assert out4.shape == sig.shape, 'Shape error in simulation output' + \
            ' with holder of order 1 and with resampling.'

    simu = SimuObj(system, nl_order_max=3)
    out5 = simu.simulation(sig, out_opt='output_by_order')
    out6 = simu.simulation(sig, out_opt='state')
    out7 = simu.simulation(sig, out_opt='state_by_order')
    assert out5.shape == (3, ) + sig.shape, 'Shape error in simulation' + \
            ' output when output_by_order is wanted.'
    assert out6.shape == (system.dim['state'],) + sig.shape, \
            'Shape error in simulation output when state is wanted.'
    assert out7.shape == (3, system.dim['state']) + sig.shape, \
            'Shape error in simulation output when state_by_order is wanted.'


    ############################
    ## Loudspeaker simulation ##
    ############################

    # Input signal
    fs = 44100
    T = 1
    f1 = 50
    f2 = 100
    amp = 10
    time_vector = np.arange(0, T, step=1/fs)
    f0_vector = np.linspace(f1, f2, num=len(time_vector))
    signal = amp * np.sin(2 * np.pi * f0_vector * time_vector)

    loudspeaker = loudspeaker_sica(output='current')
    options ={'fs': fs,
              'nl_order_max': 3,
              'holder_order': 1,
              'resampling': False}

    # Simulation
    simu1 = SimuObj(loudspeaker, **options)
    start1 = time.time()
    out1 = simu1.simulation(signal)
    end1 = time.time()

    options['holder_order'] = 0
    simu2 = SimuObj(loudspeaker, **options)
    start2 = time.time()
    out2 = simu2.simulation(signal)
    end2 = time.time()

    # Results
    plot_sig_io(signal, out1, time_vector,
                name='Input-output (holder of order 1)')
    print('Computation time (holder of order 1): {}s'.format(end1-start1))

    plot_sig_io(signal, out2, time_vector,
                name='Input-output (holder of order 0)')
    print('Computation time (holder of order 0):  {}s'.format(end2-start2))

    diff = out1 - out2
    diff.shape = diff.shape +(1,)
    plot_sig_coll(diff, time_vector, name='Difference')


    ########################
    ## Kernel computation ##
    ########################

    # Parameters
    options = {'fs': 2000,
               'nl_order_max': 2,
               'holder_order': 1}
    T = 0.03

    # Test system
    sys_simu = SimuObj(test(mode='numeric'), **options)
    t_kernels = sys_simu.compute_kernels(T, which='time')
    t_kernels, f_kernels = sys_simu.compute_kernels(T, which='both')
    f_kernels = sys_simu.compute_kernels(T, which='freq')

    # Second-order system with nonlinear damping
    damping_sys = nl_damping(gain=1, f0=100, damping=0.2, nl_coeff=[1e-1, 3e-5])
    damping_simu = SimuObj(damping_sys, **options)
    time_kernels, freq_kernels_from_time = \
                                damping_simu.compute_kernels(T, which='both')
    freq_kernels = damping_simu.compute_kernels(T, which='freq')

    N = len(time_kernels[1])
    time_vec = np.linspace(0, (N-1)/options['fs'], num=N)
    freq_vec = np.fft.fftshift(np.fft.fftfreq(N, d=1/options['fs']))

    plot_kernel_time(time_vec, time_kernels[1])
    plot_kernel_time(time_vec, time_kernels[2], style='wireframe')
    plot_kernel_freq(freq_vec, freq_kernels_from_time[1],
                     title='Transfer kernel of order 1 ' + \
                           '(computed from Volterra kernel).')
    plot_kernel_freq(freq_vec, freq_kernels_from_time[2], style='wireframe',
                     title='Transfer kernel of order 2 ' + \
                           '(computed from Volterra kernel).')
    plot_kernel_freq(freq_vec, freq_kernels[1],
                     title='Transfer kernel of order 1')
    plot_kernel_freq(freq_vec, freq_kernels[2], style='wireframe',
                     title='Transfer kernel of order 2')