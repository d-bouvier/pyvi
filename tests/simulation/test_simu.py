# -*- coding: utf-8 -*-
"""
Test script for and pyvi.simulation.simu

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 02 May 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
import time
from pyvi.system.dict import test, loudspeaker_sica
from pyvi.utilities.plotbox import plot_sig_io, plot_sig_coll


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    ## Test if simulation works correctly ##
    sig_test = np.ones((10000,))
    system_test = test(mode='numeric')
    out1 = system_test.simulation(sig_test, holder_order=0, resampling=False)
    assert out1.shape == sig_test.shape, 'Shape error in simulation output' + \
            ' with holder of order 0 and without resampling.'
    out2 = system_test.simulation(sig_test, holder_order=1, resampling=False)
    assert out1.shape == sig_test.shape, 'Shape error in simulation output' + \
            ' with holder of order 1 and without resampling.'
    out3 = system_test.simulation(sig_test, holder_order=0, resampling=True)
    assert out1.shape == sig_test.shape, 'Shape error in simulation output' + \
            ' with holder of order 0 and with resampling.'
    out4 = system_test.simulation(sig_test, holder_order=1, resampling=True)
    assert out1.shape == sig_test.shape, 'Shape error in simulation output' + \
            ' with holder of order 1 and with resampling.'
    out5 = system_test.simulation(sig_test, nl_order_max=3,
                                  out_opt='output_by_order')
    assert out5.shape == (3, ) + sig_test.shape, 'Shape error in simulation' + \
            ' output when output_by_order is wanted.'
    out6 = system_test.simulation(sig_test, nl_order_max=3, out_opt='state')
    assert out6.shape == (system_test.dim['state'],) + sig_test.shape, \
            'Shape error in simulation output when state is wanted.'
    out7 = system_test.simulation(sig_test, nl_order_max=3,
                                  out_opt='state_by_order')
    assert out7.shape == (3, system_test.dim['state']) + sig_test.shape, \
            'Shape error in simulation output when state_by_order is wanted.'

    ## Loudspeaker simulation ##
    # Input signal
    fs = 44100
    T = 1
    f1 = 50
    f2 = 100
    amp = 10
    time_vector = np.arange(0, T, step=1/fs)
    f0_vector = np.linspace(f1, f2, num=len(time_vector))
    sig = amp * np.sin(2 * np.pi * f0_vector * time_vector)
    system = loudspeaker_sica(output='current')
    options ={'fs': fs,
              'nl_order_max': 3,
              'holder_order': 1,
              'resampling': False}

    # Simulation
    start1 = time.time()
    out1 = system.simulation(sig, **options)
    end1 = time.time()

    options['holder_order'] = 0
    start2 = time.time()
    out2 = system.simulation(sig, **options)
    end2 = time.time()

    # Results
    plot_sig_io(sig, out1, time_vector, name='Input-output (holder of order 1)')
    print('Computation time (holder of order 1): {}s'.format(end1-start1))

    plot_sig_io(sig, out2, time_vector, name='Input-output (holder of order 0)')
    print('Computation time (holder of order 0):  {}s'.format(end2-start2))

    diff = out1 - out2
    diff.shape = diff.shape +(1,)
    plot_sig_coll(diff, time_vector, name='Difference')