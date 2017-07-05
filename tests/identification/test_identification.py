# -*- coding: utf-8 -*-
"""
Test script for and pyvi.identification.identification

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 23 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
import pyvi.identification.identification as identif
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
    sigma = np.sqrt(2)
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

    # Assert signal length is great enough
    nb_samples_in_kernels = binomial(M+N, N) - 1
    assert nb_samples_in_kernels <= L, '{} data samples given, '.format(L) + \
            'require at least {}'.format(nb_samples_in_kernels)


    #####################
    ## Data simulation ##
    #####################

    print('Computing data for separation ...', end=' ')
    # Ground truth simulation
    out_order_true = system4simu.simulation(input_sig,
                                            out_opt='output_by_order')

    # Data for AS separation method
    AS_method = sep.AS(N=N)
    input_coll = AS_method.gen_inputs(input_sig)
    output_coll = np.zeros(input_coll.shape)
    for ind in range(input_coll.shape[0]):
        output_coll[ind] = system4simu.simulation(input_coll[ind])
    out_order_AS = AS_method.process_outputs(output_coll)

    # Data for AS separation method
    PAS_method = sep.PAS(N=N)
    input_coll = PAS_method.gen_inputs(input_sig_cplx)
    output_coll = np.zeros(input_coll.shape)
    for ind in range(input_coll.shape[0]):
        output_coll[ind] = system4simu.simulation(input_coll[ind])
    out_order_PAS = PAS_method.process_outputs(output_coll)
    out_term_PAS = PAS_method.process_outputs(output_coll, raw_mode=True)

    print('Done.')


    ############################
    ## Kernels identification ##
    ############################

    # Initialization
    kernels = dict()
    methods_list = ['true', 'direct', 'order_true', 'order_by_AS',
                    'order_by_PAS', 'term_by_PAS']

    # Pre-computation of phi
    print('Computing Phi ...', end=' ')
    phi_orders = identif._orderKLS_construct_phi(input_sig, M, N)
    phi_terms = identif._termKLS_construct_phi(input_sig_cplx, M, N)
    print('Done.')

    # Identification
    print('Computing identification ...', end=' ')
    kernels['true'] = system4simu.compute_kernels(tau, which='time')
    kernels['direct'] = identif.KLS(input_sig, out_order_true.sum(axis=0),
                                    M, N, phi=phi_orders)
    kernels['order_true'] = identif.orderKLS(input_sig, out_order_true,
                                    M, N, phi=phi_orders)
    kernels['order_by_AS'] = identif.orderKLS(input_sig, out_order_AS,
                                    M, N, phi=phi_orders)
    kernels['order_by_PAS'] = identif.orderKLS(input_sig, out_order_PAS,
                                    M, N, phi=phi_orders)
    kernels['term_by_PAS'] = identif.termKLS(input_sig_cplx, out_term_PAS,
                                    M, N, phi=phi_terms)
    print('Done.')


    ##########################
    ## Identification error ##
    ##########################

    # Estimation error
    print('\nIdentification error (without noise)')
    print('------------------------------------')
    errors = dict()
    for method, val in kernels.items():
        errors[method] = error_measure(kernels['true'], val)
        print('{:10} :'.format(method), errors[method])
    print()


    ###################
    ## Kernels plots ##
    ###################


    print('Printing plots ...', end=' ')

    # Plots
    style2D = 'surface'
    str1 = ['Kernel of order 1 - ',  'Kernel of order 2 - ']
    title_str = {'true': 'Ground truth',
                 'direct': 'Identification on output signal',
                 'order_true': 'Identification on true orders',
                 'order_by_AS': 'Identification on orders estimated via AS',
                 'order_by_PAS': 'Identification on orders estimated via PAS',
                 'term_by_PAS': 'Identification on terms estimated via PAS'}

    for method, val in kernels.items():
        name = 'Kernel of order {} - ' + title_str.get(method, 'unknown')
        plot_kernel_time(tau_vec, val[1], title=name.format(1))
        plot_kernel_time(tau_vec, val[2], style=style2D, title=name.format(2))

    print('Done.')