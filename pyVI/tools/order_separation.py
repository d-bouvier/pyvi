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
import datetime


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
        rms_error = rms(error_sig[:,n])
        rms_ref = rms(signals_ref[:,n])
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
    return np.sqrt( np.mean(np.abs(sig)**2) )


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


def simu_collection(input_sig, system, fs=44100, N=1, hold_opt=1,
                    name='unknown', method='boyd', param={'nl_order_max' :1}):
    """
    Make collection of simulation with inputs derived from a based signal.
    (only works with SISO system)
    """

    def update_parameters():
        if method == 'boyd':
            param.update({'dtype': 'float64',
                          'K': param['nl_order_max']})
            if not 'coeff' in param:
                param['coeff'] = np.zeros((param['K']))
                for idx in range(param['K']):
                    param['coeff'][idx] = (-1)**idx * (np.pi/2)**(int(idx/2))
        elif method == 'complex':
            param.update({'dtype': 'complex128',
                          'K': param['nl_order_max']})
            if not 'w' in param:
                param['w'] = np.exp(1j * 2 * np.pi / param['K'])
            if not 'rho' in param:
                param['rho'] = 1
        return param

    def create_input_coll(input_coll):
        if method == 'boyd':
            for idx in range(param['K']):
                input_coll[idx, :] = param['coeff'][idx] * input_sig
        elif method == 'complex':
            for idx in range(param['K']):
                input_coll[idx, :] = param['rho'] * (param['w']**idx) * \
                                     input_sig
        return input_coll

    name += '_' + datetime.datetime.now().strftime('%Y_%m_%d')
    len_sig = input_sig.shape[0]
    param = update_parameters()

    # Simulation for the basic input
    out_by_order = simulation(input_sig, system, fs=fs, nl_order_max=N,
                              hold_opt=hold_opt, out='output_by_order')
    out_by_order = out_by_order[:, 0, :]
    out_by_order.dtype = param['dtype']

    # Initialization
    input_coll = create_input_coll(np.zeros((param['K'], len_sig),
                                             dtype=param['dtype']))
    output_coll = np.zeros((param['K'], len_sig), dtype=param['dtype'])

    # Simulation for the different inputs of input_coll
    for idx in range(param['K']):
        out = simulation(input_coll[idx, :], system, fs=fs,
                         nl_order_max=N, hold_opt=hold_opt, out='out')
        output_coll[idx, :] = out[:, 0]

    # Saving data
    folders = ('order_separation', name)
    simu_param = {'fs': fs,
                  'nl_order_max': N,
                  'sampler_holder_option': hold_opt}

    save_data_pickle({'sep_method': method,
                      'sep_param': param,
                      'simu_param': simu_param},
                     'config', folders)
    save_data_numpy({'input': input_sig,
                     'input_collection': input_coll,
                     'output': out_by_order.sum(1),
                     'output_by_order': out_by_order,
                     'output_collection': output_coll,
                     'time': [n / fs for n in range(len_sig)]},
                    'data', folders)
