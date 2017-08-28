# -*- coding: utf-8 -*-
"""
Test script for pyvi/identification/methods.py

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 27 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import sys
import numpy as np
import pyvi.identification.methods as identif
import pyvi.separation.methods as sep
from pyvi.identification.tools import error_measure
from pyvi.system.dict import create_nl_damping
from pyvi.simulation.simu import SimulationObject
from pyvi.utilities.mathbox import binomial
from mytoolbox.utilities.misc import my_parse_arg_for_tests


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    indent = my_parse_arg_for_tests()


    ###############################
    ## Parameters specifications ##
    ###############################

    # System specification
    f0_voulue = 200
    damping = 0.7
    system = create_nl_damping(gain=1, f0=f0_voulue/(np.sqrt(1 - damping**2)),
                               damping=damping, nl_coeff=[3, 7e-4])

    # Input signal specification
    fs = 1500
    T = 1
    sigma = 1/10
    tau = 0.006

    time_vec = np.arange(0, T, 1/fs)
    L = time_vec.shape[0]
    tau_vec = np.arange(0, tau+1/fs, 1/fs)
    M = tau_vec.shape[0]

    covariance = [[sigma, 0], [0, sigma]]
    random_sig = np.random.multivariate_normal([0, 0], covariance, size=L)
    input_sig_cplx = random_sig[:, 0] + 1j * random_sig[:, 1]
    input_sig = 2 * np.real(input_sig_cplx)

    # Simulation specification
    N = 3
    resampling = False
    system4simu = SimulationObject(system, fs=fs, nl_order_max=N,
                                   resampling=resampling)


    #####################
    ## Data simulation ##
    #####################

    # Ground truth simulation
    out_order_true = system4simu.simulation(input_sig,
                                            out_opt='output_by_order')

    # Data for PAS separation method
    PAS = sep.PAS(N=N)
    inputs_PAS = PAS.gen_inputs(input_sig_cplx)
    outputs_PAS = np.zeros(inputs_PAS.shape)
    for ind in range(inputs_PAS.shape[0]):
        outputs_PAS[ind] = system4simu.simulation(inputs_PAS[ind])
    _, term_PAS = PAS.process_outputs(outputs_PAS, raw_mode=True)

    # Data for PS separation method
    PS = sep.PS(N=N)
    inputs_PS = PS.gen_inputs(input_sig_cplx)
    outputs_PS = np.zeros(inputs_PS.shape)
    for ind in range(inputs_PS.shape[0]):
        outputs_PS[ind] = system4simu.simulation(inputs_PS[ind])
    phase_PS = PS.process_outputs(outputs_PS)


    ############################
    ## Kernels identification ##
    ############################

    # Initialization
    kernels = dict()

    # Pre-computation of phi
    phi_orders = identif._orderKLS_construct_phi(input_sig, M, N)
    phi_terms = identif._termKLS_construct_phi(input_sig_cplx, M, N)

    # Testing KLS
    message = ''
    print(indent + 'Testing KLS()...', end=' ')
    try:
        kernels['direct'] = identif.KLS(input_sig, out_order_true.sum(axis=0),
                                        M, N, phi=phi_orders)
    except:
        message += indent + (' ' * 3) + 'KLS() returned an error: ' + \
                   str(sys.exc_info()[1]) + '\n'
    print('Done.')
    print(message, end='')

    # Testing orderKLS
    message = ''
    print(indent + 'Testing orderKLS()...', end=' ')
    try:
        kernels['order'] = identif.orderKLS(input_sig, out_order_true,
                                            M, N, phi=phi_orders)
    except:
        message += indent + (' ' * 3) + 'orderKLS() returned an error: ' + \
                   str(sys.exc_info()[1]) + '\n'
    print('Done.')
    print(message, end='')

    # Testing termKLS
    message = ''
    print(indent + 'Testing termKLS()...', end=' ')
    try:
        for cast_mode in ['real', 'real-imag']:
            for mode in ['mean', 'mmse']:
                name = 'term_' + ('R' if cast_mode == 'real' else '') + mode
                kernels[name] = identif.termKLS(input_sig_cplx, term_PAS, M, N,
                                                phi=phi_terms, mode=mode,
                                                cast_mode=cast_mode)
    except:
        message += indent + (" " * 3) + "termKLS() returned an error " + \
                   "(with 'cast_mode' == " + cast_mode + "and " + \
                   "'mode' == " + mode +"): " + str(sys.exc_info()[1]) + '\n'
    print('Done.')
    print(message, end='')

    # Testing phaseKLS
    message = ''
    print(indent + 'Testing phaseKLS()...', end=' ')
    try:
        for cast_mode in ['real', 'real-imag']:
            name = 'phase' + ('_R' if cast_mode == 'real' else '')
            kernels[name] = identif.phaseKLS(input_sig_cplx, phase_PS, M, N,
                                             phi=phi_terms,
                                             cast_mode=cast_mode)
    except:
        message += indent + (" " * 3) + "phaseKLS() returned an error " + \
                   "(with 'cast_mode' == " + cast_mode + "): " + \
                   str(sys.exc_info()[1]) + '\n'
    print('Done.')
    print(message, end='')

    # Testing iterKLS
    message = ''
    print(indent + 'Testing iterKLS()...', end=' ')
    try:
        for cast_mode in ['real', 'real-imag']:
            name = 'iter' + ('_R' if cast_mode == 'real' else '')
            kernels[name] = identif.iterKLS(input_sig_cplx, phase_PS, M, N,
                                             phi=phi_terms,
                                             cast_mode=cast_mode)
    except:
        message += indent + (" " * 3) + "iterKLS() returned an error " + \
                   "(with 'cast_mode' == " + cast_mode + "): " + \
                   str(sys.exc_info()[1]) + '\n'
    print('Done.')
    print(message, end='')
