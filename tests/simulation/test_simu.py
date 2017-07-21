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
import argparse
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

    #####################
    ## Parsing options ##
    #####################

    parser = argparse.ArgumentParser()
    parser.add_argument('-ind', '--indentation', type=int, default=0)
    args = parser.parse_args()
    indent = args.indentation
    ss = ' ' * indent


    #########################
    ## Method simulation() ##
    #########################

    sig = np.ones((10000,))
    system = create_test(mode='numeric')

    print(ss + 'Testing SimulationObject.simulation()...', end=' ')

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


    ##############################
    ## Method compute_kernels() ##
    ##############################

    # Parameters
    options = {'fs': 2000,
               'nl_order_max': 2,
               'holder_order': 1}
    T = 0.03

    # Test system
    print(ss + 'Testing SimulationObject.compute_kernels()...', end=' ')
    sys_simu = SimuObj(create_test(mode='numeric'), **options)
    t_kernels = sys_simu.compute_kernels(T, which='time')
    t_kernels, f_kernels = sys_simu.compute_kernels(T, which='both')
    f_kernels = sys_simu.compute_kernels(T, which='freq')
    print('Done.')
