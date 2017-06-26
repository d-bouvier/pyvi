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

    #################################
    ### Parameters specifications ###
    #################################

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

    time_vector = np.arange(0, T, 1/fs)
    L = time_vector.shape[0]
    tau_vector = np.arange(0, tau+1/fs, 1/fs)
    M = tau_vector.shape[0]

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


    #######################
    ### Data simulation ###
    #######################

    # Ground truth simulation
    output_sig_by_order = system4simu.simulation(input_sig,
                                                 out_opt='output_by_order')
    output_sig = np.sum(output_sig_by_order, axis=0)


    ##############################
    ### Kernels identification ###
    ##############################

    # Initialization
    kernels = dict()
    methods_list = ['true', 'direct', 'orders']

    # Identification
    kernels['true'] = system4simu.compute_kernels(tau, which='time')
    kernels['direct'] = identif.KLS(input_sig, output_sig_by_order.sum(axis=0),
                                    M, N)
    kernels['orders'] = identif.orderKLS(input_sig, output_sig_by_order, M, N)


    ############################
    ### Identification error ###
    ############################

    # Estimation error
    print('Identification error (without noise)')
    print('------------------------------------')
    errors = dict()
    for method in methods_list:
        errors[method] = error_measure(kernels['true'], kernels[method])
        print('{:10} :'.format(method), errors[method])


    #####################
    ### Kernels plots ###
    #####################

    # Plots
    style2D = 'surface'
    str1 = ['Kernel of order 1 - ',  'Kernel of order 2 - ']
    title_str = {'true': 'Ground truth',
                 'direct' : 'Identification directly on output signal',
                 'order' : 'Identification directly on nonlinear orders'}

    for method in methods_list:
        name = 'Kernel of order {} - ' + title_str.get(method, 'unknown')
        plot_kernel_time(tau_vector, kernels[method][1], title=name.format(1))
        plot_kernel_time(tau_vector, kernels[method][2], style=style2D,
                         title=name.format(2))