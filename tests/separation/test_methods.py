# -*- coding: utf-8 -*-
"""
Test script for and pyvi.separation.order_separation

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 12 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
import pyvi.separation.methods as sep
from pyvi.system.dict import create_nl_damping
from pyvi.simulation.simu import SimulationObject
from pyvi.utilities.plotbox import plot_sig, plot_coll
import matplotlib.pyplot as plt


#==============================================================================
# Main script
#==============================================================================


if __name__ == '__main__':
    """
    Main script for testing.
    """

    print()

    ################
    ## Parameters ##
    ################

    fs = 4410
    T = 0.2
    time_vec = np.arange(0, T, 1/fs)
    f1 = 100
    f2 = 133
    input_cplx = np.exp(2j * np.pi * f1 * time_vec) + \
                 np.exp(2j * np.pi * f2 * time_vec)
    input_real = 2 * np.real(input_cplx)

    nl_order_max = 3
    system = SimulationObject(create_nl_damping(nl_coeff=[1e-1, 3e-5]), fs=fs,
                              nl_order_max=nl_order_max)


    ################
    ## Separation ##
    ################

    order_cplx = system.simulation(input_cplx, out_opt='output_by_order')
    order = system.simulation(input_real, out_opt='output_by_order')

    print('Testing _PS method ...', end=' ')
    _PS = sep._PS(N=nl_order_max)
    input_coll = _PS.gen_inputs(input_cplx)
    output_coll = np.zeros(input_coll.shape, dtype='complex128')
    for ind in range(input_coll.shape[0]):
        output_coll[ind] = system.simulation(input_coll[ind])
    order_est_PS = _PS.process_outputs(output_coll)
    assert np.allclose(order_cplx, order_est_PS), \
                'Separation error in _PS method.'
    print('Done.')

    print('Testing AS method ...', end=' ')
    AS = sep.AS(N=nl_order_max)
    input_coll = AS.gen_inputs(input_real)
    output_coll = np.zeros(input_coll.shape)
    for ind in range(input_coll.shape[0]):
        output_coll[ind] = system.simulation(input_coll[ind])
    order_est_AS = AS.process_outputs(output_coll)
    assert np.allclose(order, order_est_AS), 'Separation error in AS method.'
    print('Done.')

    print('Testing PS method ...', end=' ')
    PS = sep.PS(N=nl_order_max)
    input_coll = PS.gen_inputs(input_cplx)
    output_coll = np.zeros(input_coll.shape)
    for ind in range(input_coll.shape[0]):
        output_coll[ind] = system.simulation(input_coll[ind])
    sig_est_PS = PS.process_outputs(output_coll)
    assert np.allclose(order_cplx[nl_order_max-2:],
                       sig_est_PS[nl_order_max-1:nl_order_max+1]), \
                'Separation error in PS method.'
    print('Done.')

    print('Testing PAS method ...', end=' ')
    PAS = sep.PAS(N=nl_order_max)
    input_coll = PAS.gen_inputs(input_cplx)
    output_coll = np.zeros(input_coll.shape)
    for ind in range(input_coll.shape[0]):
        output_coll[ind] = system.simulation(input_coll[ind])
    order_est_PAS = PAS.process_outputs(output_coll)
    term_est_PAS = PAS.process_outputs(output_coll, raw_mode=True)
    assert np.allclose(order, order_est_PAS), 'Separation error in PAS method.'
    print('Done.')

    print('Testing PAS_v2 method ...', end=' ')
    PAS_v2 = sep.PAS_v2(N=nl_order_max)
    input_coll = PAS_v2.gen_inputs(input_cplx)
    output_coll = np.zeros(input_coll.shape)
    for ind in range(input_coll.shape[0]):
        output_coll[ind] = system.simulation(input_coll[ind])
    order_est_PASv2 = PAS_v2.process_outputs(output_coll)
    term_est_PASv2 = PAS_v2.process_outputs(output_coll, raw_mode=True)
    assert np.allclose(order, order_est_PASv2), \
                'Separation error in PASv2 method.'
    print('Done.')



    ##################
    ## Signal plots ##
    ##################

    title = 'Method {} - True orders, estimated orders and errors.'
    title_order = ['Order {}'.format(n) for n in range(1, nl_order_max+1)]
    title_type = ['True orders', 'Estimated orders', 'Error']
    title_phase = ['Phase {}'.format(n) for n in range(nl_order_max+1)]

    print('Printing plots ...', end=' ')

    plot_coll(time_vec, (np.real(order_cplx), np.real(order_est_PS),
                         np.real(order_cplx - order_est_PS)),
              title=title.format('_PS'), xtitle=title_type, ytitle=title_order)

    plot_coll(time_vec, (order, order_est_AS, order - order_est_AS),
              title=title.format('AS'), xtitle=title_type, ytitle=title_order)

    plot_sig(time_vec, sig_est_PS[:nl_order_max+1],
             title='Method PS - Estimated terms',
             title_plots=title_phase)

    plot_coll(time_vec, (order, order_est_PAS, order - order_est_PAS),
              title=title.format('PAS'), xtitle=title_type, ytitle=title_order)

    plt.figure('Method raw-PAS - Estimated terms')
    plt.clf()
    nb_col = 2*(nl_order_max+1)
    shape = (nl_order_max, nb_col)
    for n in range(1, nl_order_max+1):
        for q in range(0, n+1):
            pos = (n-1, nl_order_max - n + 2*q)
            ax = plt.subplot2grid(shape, pos, colspan=2)
            ax.plot(time_vec, np.real(term_est_PAS[(n, q)]), 'b')
            ax.plot(time_vec, np.imag(term_est_PAS[(n, q)]), 'r')

    plot_coll(time_vec, (order, order_est_PASv2, order - order_est_PASv2),
              title=title.format('PAS_v2'), xtitle=title_type,
              ytitle=title_order)

    plt.figure('Method raw-PAS_v2 - Estimated terms')
    plt.clf()
    nb_col = 2*(nl_order_max+1)
    shape = (nl_order_max, nb_col)
    for n in range(1, nl_order_max+1):
        for q in range(0, n+1):
            pos = (n-1, nl_order_max - n + 2*q)
            ax = plt.subplot2grid(shape, pos, colspan=2)
            ax.plot(time_vec, np.real(term_est_PAS[(n, q)]), 'b')
            ax.plot(time_vec, np.imag(term_est_PAS[(n, q)]), 'r')

    print('Done.')