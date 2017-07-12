# -*- coding: utf-8 -*-
"""
Test script for pyvi.identification.methods

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 12 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
import pyvi.identification.methods as identif
import pyvi.separation.methods as sep
from pyvi.identification.tools import error_measure
from pyvi.system.dict import create_nl_damping
from pyvi.simulation.simu import SimulationObject
from pyvi.utilities.plotbox import plot_kernel_time
from pyvi.utilities.mathbox import binomial


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    print()

    ###############################
    ## Parameters specifications ##
    ###############################

    # System specification
    f0_voulue = 200
    damping = 0.7
    system = create_nl_damping(gain=1, f0=f0_voulue/(np.sqrt(1 - damping**2)),
                               damping=damping, nl_coeff=[3, 7e-4])

    # Input signal specification
    fs = 3000
    T = 2
    sigma = 1/10
    tau = 0.005

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
    system4simu = SimulationObject(system, fs=fs, nl_order_max=N)
    snr = -50

    # Assert signal length is great enough
    nb_samples_in_kernels = binomial(M+N, N) - 1
    assert nb_samples_in_kernels <= L, '{} data samples given, '.format(L) + \
            'require at least {}'.format(nb_samples_in_kernels)


    #####################
    ## Data simulation ##
    #####################

    print('Computing data for separation...', end=' ')
    # Ground truth simulation
    out_order_true = system4simu.simulation(input_sig,
                                            out_opt='output_by_order')

    # Data for AS separation method
    AS = sep.AS(N=N, gain=0.65)
    inputs_AS = AS.gen_inputs(input_sig)
    outputs_AS = np.zeros(inputs_AS.shape)
    for ind in range(inputs_AS.shape[0]):
        outputs_AS[ind] = system4simu.simulation(inputs_AS[ind])
    order_AS = AS.process_outputs(outputs_AS)

    # Data for PAS separation method
    PAS = sep.PAS(N=N, gain=0.65)
    inputs_PAS = PAS.gen_inputs(input_sig_cplx)
    outputs_PAS = np.zeros(inputs_PAS.shape)
    for ind in range(inputs_PAS.shape[0]):
        outputs_PAS[ind] = system4simu.simulation(inputs_PAS[ind])
    order_PAS, term_PAS = PAS.process_outputs(outputs_PAS, raw_mode=True)

    # Data for PS separation method
    PS = sep.PS(N=N)
    inputs_PS = PS.gen_inputs(input_sig_cplx)
    outputs_PS = np.zeros(inputs_PS.shape)
    for ind in range(inputs_PS.shape[0]):
        outputs_PS[ind] = system4simu.simulation(inputs_PS[ind])
    phase_PS = PS.process_outputs(outputs_PS)

    # Noisy data
    amp_max = max(np.max(np.abs(order_AS)), np.max(np.abs(order_PAS)))
    amp_noise = amp_max * 10**(snr/20)
    shape_max = max(outputs_AS.shape, outputs_PAS.shape)
    noise = np.random.normal(scale=amp_noise, size=shape_max)

    nb_AS = outputs_AS.shape[0]
    nb_PAS = outputs_PAS.shape[0]
    nb_PS = outputs_PS.shape[0]
    out_order_n = out_order_true + noise[:N]
    order_AS_n = AS.process_outputs(outputs_AS + noise[:nb_AS])
    order_PAS_n, term_PAS_n = PAS.process_outputs(outputs_PAS + noise[:nb_PAS],
                                                  raw_mode=True)
    phase_PS_n = PS.process_outputs(outputs_PS + noise[:nb_PS])
    print('Done.')


    ############################
    ## Kernels identification ##
    ############################

    # Initialization
    kernels = dict()
    kernels_n = dict()

    # Pre-computation of phi
    print('Computing phi...', end=' ')
    phi_orders = identif._orderKLS_construct_phi(input_sig, M, N)
    phi_terms = identif._termKLS_construct_phi(input_sig_cplx, M, N)
    print('Done.')

    # Identification (on clean data)
    print('Computing identification (on clean data)...', end=' ')
    kernels['true'] = system4simu.compute_kernels(tau, which='time')
    kernels['direct'] = identif.KLS(input_sig, out_order_true.sum(axis=0),
                                    M, N, phi=phi_orders)
    kernels['order_true'] = identif.orderKLS(input_sig, out_order_true, M, N,
                                             phi=phi_orders)
    kernels['order_AS'] = identif.orderKLS(input_sig, order_AS, M, N,
                                           phi=phi_orders)
    kernels['order_PAS'] = identif.orderKLS(input_sig, order_PAS, M, N,
                                            phi=phi_orders)
    kernels['term_Rmean'] = identif.termKLS(input_sig_cplx, term_PAS, M, N,
                                            phi=phi_terms, cast_mode='real',
                                            mode='mean')
    kernels['term_Rmmse'] = identif.termKLS(input_sig_cplx, term_PAS, M, N,
                                            phi=phi_terms, cast_mode='real')
    kernels['term_mean'] = identif.termKLS(input_sig_cplx, term_PAS, M, N,
                                           phi=phi_terms, mode='mean')
    kernels['term_mmse'] = identif.termKLS(input_sig_cplx, term_PAS, M, N,
                                           phi=phi_terms)
    kernels['phase_R'] = identif.phaseKLS(input_sig_cplx, phase_PS, M, N,
                                          phi=phi_terms, cast_mode='real')
    kernels['phase'] = identif.phaseKLS(input_sig_cplx, phase_PS, M, N,
                                        phi=phi_terms)
    kernels['iter_R'] = identif.iterKLS(input_sig_cplx, phase_PS, M, N,
                                        phi=phi_terms, cast_mode='real')
    kernels['iter'] = identif.iterKLS(input_sig_cplx, phase_PS, M, N,
                                      phi=phi_terms)
    print('Done.')

    # Identification (on noisy data)
    print('Computing identification (on noisy data)...', end=' ')
    kernels_n['direct'] = identif.KLS(input_sig, out_order_n.sum(axis=0),
                                      M, N, phi=phi_orders)
    kernels_n['order_true'] = identif.orderKLS(input_sig, out_order_n, M, N,
                                               phi=phi_orders)
    kernels_n['order_AS'] = identif.orderKLS(input_sig, order_AS_n, M, N,
                                             phi=phi_orders)
    kernels_n['order_PAS'] = identif.orderKLS(input_sig, order_PAS_n, M, N,
                                              phi=phi_orders)
    kernels_n['term_Rmean'] = identif.termKLS(input_sig_cplx, term_PAS_n, M, N,
                                              phi=phi_terms, cast_mode='real',
                                              mode='mean')
    kernels_n['term_Rmmse'] = identif.termKLS(input_sig_cplx, term_PAS_n, M, N,
                                              phi=phi_terms, cast_mode='real')
    kernels_n['term_mean'] = identif.termKLS(input_sig_cplx, term_PAS_n, M, N,
                                             phi=phi_terms, mode='mean')
    kernels_n['term_mmse'] = identif.termKLS(input_sig_cplx, term_PAS_n, M, N,
                                             phi=phi_terms)
    kernels_n['phase_R'] = identif.phaseKLS(input_sig_cplx, phase_PS_n, M,
                                            N, phi=phi_terms, cast_mode='real')
    kernels_n['phase'] = identif.phaseKLS(input_sig_cplx, phase_PS_n, M, N,
                                          phi=phi_terms)
    kernels_n['iter_R'] = identif.iterKLS(input_sig_cplx, phase_PS_n, M, N,
                                          phi=phi_terms, cast_mode='real')
    kernels_n['iter'] = identif.iterKLS(input_sig_cplx, phase_PS_n, M, N,
                                        phi=phi_terms)
    print('Done.')


    ###################
    ## Kernels plots ##
    ###################

    print('Printing plots...', end=' ')

    # Plots
    style2D = 'surface'
    str1 = ['Kernel of order 1 - ',  'Kernel of order 2 - ']
    title_str = {'true': 'Ground truth',
                 'direct': 'Identification on output signal',
                 'order_true': 'Identification on true orders',
                 'order_AS': 'Identification on orders estimated via AS',
                 'order_PAS': 'Identification on orders estimated via PAS',
                 'term_Rmean': 'Identification on terms estimated via PAS' + \
                               " (only real part used and 'mean' mode used)",
                 'term_Rmmse': 'Identification on terms estimated via PAS' + \
                               " (only real part used and 'mmse' mode used)",
                 'term_mean': 'Identification on terms estimated via PAS' + \
                              " ('mean' mode used)",
                 'term_mmse': 'Identification on terms estimated via PAS' + \
                               " ('mmse' mode used)",
                 'phase_R': 'Identification on phase signals estimated' + \
                            ' via PS (only real part used)',
                 'phase': 'Identification on phase signals estimated via PS',
                 'iter': 'Recursive identification on phase signals' + \
                         ' estimated via PS',
                 'iter_R': 'Recursive identification on phase signals ' + \
                           ' estimated via PS (only real part used)'}

    for method, val in kernels.items():
        name = 'Kernel of order {} - ' + title_str.get(method, 'Unknown method')
        plot_kernel_time(tau_vec, val[1], title=name.format(1))
        plot_kernel_time(tau_vec, val[2], style=style2D, title=name.format(2))

    print('Done.')


    ##########################
    ## Identification error ##
    ##########################

    # Estimation error (without noise)
    print('\nIdentification error (without noise)')
    print('------------------------------------')
    errors = dict()
    for method, val in kernels.items():
        errors[method] = error_measure(kernels['true'], val)
        print('{:11} :'.format(method), errors[method])

    # Estimation error (without noise)
    print('\nIdentification error (with noise)')
    print('---------------------------------')
    errors_n = dict()
    for method, val in kernels_n.items():
        errors_n[method] = error_measure(kernels['true'], val)
        print('{:11} :'.format(method), errors_n[method])
    print()