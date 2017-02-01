# -*- coding: utf-8 -*-
"""
Toolbox for nonlinear order separation.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 3 Nov. 2016
Developed for Python 3.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from pyvi.simulation.simulation import simulation
from pyvi.tools.paths import save_data_pickle, save_data_numpy


#==============================================================================
# Functions
#==============================================================================

def estimation_measure(signals_ref, signals_est, mode='default'):
    """
    Measure the estimation of a signal, in dB. The lower the value returned is,
    the better the estimation is. If the signal and its estimation are equal,
    this function returns - np.Inf.
    """
    nb_sig = signals_est.shape[1]
    error_sig = np.abs(signals_ref - signals_est)
    error_measure = []

    for n in range(nb_sig):
        rms_error = rms(error_sig)
        rms_ref = rms(signals_ref)
        if mode == 'default':
            val = safe_db(rms_error, rms_ref)
        elif mode == 'normalized':
            val = safe_db(rms_ref, rms_ref + rms_error)
        error_measure.append(val)

    return error_measure


def rms(sig):
    """
    Computation of the root-mean-square of a vector.
    """
    return np.sqrt( np.mean(sig**2) )


def safe_db(num, den):
    """
    dB computation with verification that neither the denominator or numerator
    are equal to zero.
    """
    if den == 0:
        return np.Inf
    if num == 0:
        return - np.Inf
    return 20 * np.log10(num / den)


def simu_collection(input_sig, coll_factor, system, fs=44100, N=1, hold_opt=1,
                    dtype='float', name=''):
    """
    Make collection of simulation with inputs derived from a based signal.
    """

    if name != '':
        name += '_'

    input_one_dimensional = system.dim['input'] == 1

    K = len(coll_factor)
    if input_one_dimensional:
        len_sig = input_sig.shape[0]
    else:
        len_sig = input_sig.shape[1]

    out_by_order = simulation(input_sig, system, fs=fs, nl_order_max=N,
                              hold_opt=hold_opt, out='output_by_order')
    out_by_order.dtype = dtype

    if input_one_dimensional:
        out_by_order = out_by_order[:, 0, :]
        output = np.zeros((len_sig, K), dtype=dtype)
    else:
        output = np.zeros((len_sig, system.dim['input'], K), dtype=dtype)

    for idx in range(K):
        out = simulation(input_sig * coll_factor[idx], system, fs=fs,
                         nl_order_max=N, hold_opt=hold_opt, out='out')

        if input_one_dimensional:
            output[:, idx] = out[:, 0]
        else:
            output[:, :, idx] = out

    folders = ('order_separation', 'simu_data')
    save_data_pickle({'constrast_factor': coll_factor,
                      'number_test': K,
                      'fs': fs,
                      'nonlinear_order_max': N,
                      'sampler_holder_option': hold_opt},
                     name + '{}_config', folders)
    save_data_numpy({'input': input_sig,
                     'output': out_by_order.sum(1),
                     'output_by_order': out_by_order,
                     'output_collection': output,
                     'time': [n / fs for n in range(len_sig)]},
                    name + '{}_data', folders)

    return output, out_by_order
