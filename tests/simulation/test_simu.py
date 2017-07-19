# -*- coding: utf-8 -*-
"""
Test script for pyvi.simulation.simu

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 19 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import time
import numpy as np
from pyvi.system.dict import (create_test, create_loudspeaker_sica,
                              create_nl_damping)
from pyvi.simulation.simu import SimulationObject as SimuObj
from pyvi.utilities.plotbox import (plot_sig_io, plot_sig, plot_kernel_time,
                                    plot_kernel_freq, plot_spectrogram)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    print()

    ########################################
    ## Test if simulation works correctly ##
    ########################################

    sig = np.ones((10000,))
    system = create_test(mode='numeric')

    print('Testing correctness of output shape ...', end=' ')

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

    print('Done.')


    ############################
    ## Loudspeaker simulation ##
    ############################

    # Input signal
    fs = 2000
    T = 1
    f1 = 50
    f2 = 500
    amp = 10
    time_vector = np.arange(0, T, step=1/fs)
    k = (f2 -f1)/T
    phi = 2*np.pi * (f1*time_vector + (k/2)*time_vector**2)
    signal = amp * np.sin(phi)

    loudspeaker = create_loudspeaker_sica(output='current')
    options ={'fs': fs,
              'nl_order_max': 3,
              'holder_order': 0,
              'resampling': False}

    # Simulation
    print('Computing loudspeaker simulation ...', end=' ')

    simu_0_f = SimuObj(loudspeaker, **options)
    start_0_f = time.time()
    out_0_f = simu_0_f.simulation(signal)
    end_0_f = time.time()

    options['resampling'] = True
    simu_0_t = SimuObj(loudspeaker, **options)
    start_0_t = time.time()
    out_0_t = simu_0_t.simulation(signal)
    end_0_t = time.time()

    options['holder_order'] = 1
    simu_1_t = SimuObj(loudspeaker, **options)
    start_1_t = time.time()
    out_1_t = simu_1_t.simulation(signal)
    end_1_t = time.time()

    options['resampling'] = False
    simu_1_f = SimuObj(loudspeaker, **options)
    start_1_f = time.time()
    out_1_f = simu_1_f.simulation(signal)
    end_1_f = time.time()
    print('Done.')

    # Results
    plot_sig_io(time_vector, signal, out_0_f,
                title='Input-output (holder of order 0 without resampling)')
    print('Computation time (holder of order 0 without resampling):',
          '{}s'.format(end_0_f-start_0_f))
    plot_sig_io(time_vector, signal, out_0_t,
                title='Input-output (holder of order 0 with resampling)')
    print('Computation time (holder of order 0 with resampling)   :',
          '{}s'.format(end_0_t-start_0_t))
    plot_sig_io(time_vector, signal, out_1_f,
                title='Input-output (holder of order 1 without resampling)')
    print('Computation time (holder of order 1 without resampling):',
          '{}s'.format(end_1_f-start_1_f))
    plot_sig_io(time_vector, signal, out_1_t,
                title='Input-output (holder of order 1 with resampling)')
    print('Computation time (holder of order 1 with resampling)   :',
          '{}s'.format(end_1_t-start_1_t))

    diff = np.zeros((4, len(time_vector)))
    diff[0] = out_0_f - out_0_t
    diff[1] = out_1_f - out_1_t
    diff[2] = out_0_f - out_1_f
    diff[3] = out_0_t - out_1_t
    plot_sig(time_vector, diff, title='Differences between simulation',
             title_plots=['Holder of order 1, w/ and w/o resampling',
                          'Holder of order 0, w/ and w/o resampling',
                          'Holder of order 0 and 1, w/o resampling',
                          'Holder of order 0 and 1, w/ resampling'])
    opt = {'fs': fs, 'nperseg': 128, 'noverlap': 96, 'nfft': 1024}
    plot_spectrogram(signal, title='Input spectrogram', **opt)
    plot_spectrogram(out_1_f, title='Output spectrogram with holder of ' + \
                     'order 1 and no resampling', **opt)
    plot_spectrogram(out_1_t, title='Output spectrogram with holder of ' + \
                     'order 1 and resampling', **opt)


    ########################
    ## Kernel computation ##
    ########################

    # Parameters
    options = {'fs': 2000,
               'nl_order_max': 2,
               'holder_order': 1}
    T = 0.03

    # Test system
    print('Testing kernel computation...', end=' ')
    sys_simu = SimuObj(create_test(mode='numeric'), **options)
    t_kernels = sys_simu.compute_kernels(T, which='time')
    t_kernels, f_kernels = sys_simu.compute_kernels(T, which='both')
    f_kernels = sys_simu.compute_kernels(T, which='freq')
    print('Done.')

    # Second-order system with nonlinear damping
    print('Computing kernels...', end=' ')
    damping_sys = create_nl_damping(gain=1, f0=100, damping=0.2,
                                    nl_coeff=[1e-1, 3e-5])
    damping_simu = SimuObj(damping_sys, **options)
    time_kernels, freq_kernels_from_time = \
                                damping_simu.compute_kernels(T, which='both')
    freq_kernels = damping_simu.compute_kernels(T, which='freq')
    options['resampling'] = True
    damping_simu_2 = SimuObj(damping_sys, **options)
    time_kernels_2, freq_kernels_from_time_2 = \
                                damping_simu_2.compute_kernels(T, which='both')
    freq_kernels_2 = damping_simu_2.compute_kernels(T, which='freq')
    print('Done.')

    print("Checking equality with 'resample' mode on or off...", end=' ')
    assert np.all(time_kernels[1] == time_kernels_2[1]), "Error in kernels" + \
        " computation with 'resample' mode on."
    assert np.all(time_kernels[2] == time_kernels_2[2]), "Error in kernels" + \
        " computation with 'resample' mode on."
    assert np.all(freq_kernels_from_time[1] == freq_kernels_from_time_2[1]), \
        "Error in kernels computation with 'resample' mode on."
    assert np.all(freq_kernels_from_time[2] == freq_kernels_from_time_2[2]), \
        "Error in kernels computation with 'resample' mode on."
    print('Done.')

    print('Plotting kernels...', end=' ')
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
    print('Done.')