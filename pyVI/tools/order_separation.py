# -*- coding: utf-8 -*-
"""
Toolbox for nonlinear order separation.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 3 Nov. 2016
Developed for Python 3.5.1
Uses:
 - numpy 1.11.1
 - scipy 0.18.0
 - pyvi 0.1
 - datetime
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from scipy import fftpack
from scipy.special import binom as binomial
from pyvi.simulation.simulation import simulation
from pyvi.tools.mathbox import rms, safe_db
from pyvi.tools.paths import save_data_pickle, save_data_numpy
from datetime import datetime


#==============================================================================
# Functions
#==============================================================================

def estimation_measure(signals_ref, signals_est, mode='default'):
    """
    Measure the estimation of a signal, in dB. The lower the value returned is,
    the better the estimation is. If the signal and its estimation are equal,
    this function returns - np.Inf.
    """
    nb_sig = signals_est.shape[0]
    error_sig = np.abs(signals_ref - signals_est)
    error_measure = []

    for n in range(nb_sig):
        rms_error = rms(error_sig[n])
        rms_ref = rms(signals_ref[n])
        if mode == 'default':
            val = safe_db(rms_error, rms_ref)
        elif mode == 'normalized':
            val = safe_db(rms_ref, rms_ref + rms_error)
        error_measure.append(val)

    return error_measure


def simu_collection(input_sig, system, fs=44100, hold_opt=1,
                    name='unknown', method='boyd', param={'nl_order_max' :1},
                    save_bool=False):
    """
    Make collection of simulation with inputs derived from a based signal.
    (only works with SISO system)
    """

    ## Parameters specification, in function of the separation method

    # Amplitude method (using Vandermonde matrix)
    if method == 'boyd':
        def update_parameters():
            param.update({'dtype': 'float64'})
            if not 'K' in param:
                param['K'] = param['nl_order_max']
            if not 'coeff' in param:
                vec = np.arange(param['K'])
                if not 'gain' in param:
                    param['gain'] = np.pi/2
                if param.pop('negative_gain', False):
                    param['coeff'] = (-1)**vec * param['gain']**(vec//2)
                else:
                    param['coeff'] = param['gain']**vec
            return param
        def create_input_coll(input_coll):
            for idx in range(param['K']):
                input_coll[idx, :] = param['coeff'][idx] * input_sig
            return input_coll

    # Phase method (using DFT matrix and complex signals)
    elif method == 'complex':
        def update_parameters():
            param.update({'dtype': 'complex128',
                          'K': param['nl_order_max']})
            param['w'] = np.exp(- 1j * 2 * np.pi / param['K'])
            if not 'rho' in param:
                param['rho'] = 1
            return param
        def create_input_coll(input_coll):
            for idx in range(param['K']):
                input_coll[idx, :] = param['rho'] * (param['w']**idx) * \
                                     input_sig
            return input_coll

    # Phase + amplitude method (using DFT and Vandermonde matrix)
    elif method == 'phase+amp':
        def update_parameters():
            param.update({'dtype': 'float64'})
            N = param['nl_order_max']
            param['K_phase'] = 2*N + 1
            param['K_amp'] = (N + 1) // 2
            param['K'] = param['K_phase'] * param['K_amp']
            param['nb_term'] = (N * (N + 3)) // 2
            if not 'coeff' in param:
                vec = np.arange(param['K_amp'])
                if not 'gain' in param:
                    param['gain'] = np.pi/2
                param['coeff'] = param['gain']**vec
            param['w'] = np.exp(- 1j * 2 * np.pi / param['K_phase'])
            if not 'output' in param:
                param['output'] = 'orders'
            if not 'out_type' in param:
                param['out_type'] = 'array'
            return param
        def create_input_coll(input_coll):
            for idx_amp in range(param['K_amp']):
                for idx_phase in range(param['K_phase']):
                    idx = idx_amp * param['K_phase'] + idx_phase
                    input_coll[idx, :] = 2 * np.real(param['coeff'][idx_amp] * \
                                                     param['w']**idx_phase * \
                                                     input_sig)
            return input_coll

    # Parameters initialization
    len_sig = input_sig.shape[0]
    param = update_parameters()

    ## Data simulation

    # Simulation of ground truth
    if method == 'phase+amp':
        out_by_order = simulation(np.real(input_sig), system, fs=fs,
                                  nl_order_max=param['nl_order_max'],
                                  hold_opt=hold_opt, out='output_by_order')
        out_by_order.dtype = param['dtype']
        out_by_order_bis = simulation(input_sig, system, fs=fs,
                                      nl_order_max=param['nl_order_max'],
                                      hold_opt=hold_opt, out='output_by_order')

    else:
        out_by_order = simulation(input_sig, system, fs=fs,
                                  nl_order_max=param['nl_order_max'],
                                  hold_opt=hold_opt, out='output_by_order')
        out_by_order.dtype = param['dtype']

    # Creation of input collection
    input_coll = create_input_coll(np.zeros((param['K'], len_sig),
                                             dtype=param['dtype']))

    # Simulation of the output collection
    output_coll = np.zeros((param['K'], len_sig), dtype=param['dtype'])
    for idx in range(param['K']):
        out = simulation(input_coll[idx, :], system, fs=fs,
                         nl_order_max=param['nl_order_max'],
                         hold_opt=hold_opt, out='output')
        output_coll[idx, :] = out

    # Data saving and function output
    folders = ('order_separation', name)
    simu_param = {'fs': fs,
                  'nl_order_max': param['nl_order_max'],
                  'sampler_holder_option': hold_opt}

    config = {'sep_method': method,
              'sep_param': param,
              'simu_param': simu_param,
              'time': datetime.now().strftime('%d_%m_%Y_%Hh%M')}
    data = {'input': input_sig,
            'input_collection': input_coll,
            'output': out_by_order.sum(0),
            'output_by_order': out_by_order,
            'output_collection': output_coll,
            'time': [n / fs for n in range(len_sig)]}
    if method == 'phase+amp':
        data.update({'output_by_order_cplx': out_by_order_bis,
                     'input_real': 2 * np.real(input_sig)})

    if save_bool:
        save_data_pickle(config, 'config', folders)
        save_data_numpy(data, 'data', folders)

    return data, config


def order_separation(output_coll, method, param):
    """
    Make separation of nonlinear order for a given method.
    (only works with SISO system)
    """

    ## Parameters specification, in function of the separation method

    # Amplitude method (using Vandermonde matrix)
    if method == 'boyd':
        mixing_mat = np.vander(param['coeff'], N=param['nl_order_max']+1,
                               increasing=True)[:, 1::]
        if mixing_mat.shape[0] == mixing_mat.shape[1]: # Square matrix
            return np.dot(np.linalg.inv(mixing_mat), output_coll)
        else: # Npn-square matrix (pseudo-inverse)
            return np.dot(np.linalg.pinv(mixing_mat), output_coll)

    # Phase method (using DFT matrix and complex signals)
    elif method == 'complex':
        estimation = fftpack.ifft(output_coll, n=param['nl_order_max'], axis=0)
        demixing_vec = np.vander([1/param['rho']], N=param['K'],
                                 increasing=True)
        return demixing_vec.T * np.roll(estimation, -1, axis=0)

    # Phase + amplitude method (using DFT and Vandermonde matrix)
    elif method == 'phase+amp':
        # Initialization
        len_sig = output_coll.shape[-1]
        out_per_phase = np.zeros((param['K_amp'], param['K_phase'], len_sig),
                                 dtype='complex128')
        term_combinatoric = np.zeros((param['nb_term'], len_sig),
                                      dtype='complex128')
        mixing_mat = np.vander(param['coeff'], N=param['nl_order_max']+1,
                               increasing=True)[:, 1::]

        # Inverse DFT for each set with same amplitude
        for idx in range(0, param['K_amp']):
            start = idx*param['K_phase']
            end = start + param['K_phase']
            out_per_phase[idx,:,:] = fftpack.ifft(output_coll[start:end, :],
                                                  n=param['K_phase'], axis=0)
        if param['output'] == 'raw':
            return out_per_phase[0, :, :]

        # Computation of indexes and necessary vector
        k_vec = np.arange(0, param['nb_term'])
        n_vec = (np.sqrt(9 + 8*k_vec) - 1)//2
        q_vec = k_vec + 1 - (n_vec*(n_vec+1))//2
        phase_vec = (n_vec - 2*q_vec) % param['K_phase']

        tmp = (np.arange(param['nl_order_max'], 0, -1) + 1) // 2
        nb_term = np.append(np.concatenate((tmp, tmp[::-1])), tmp[1])[::-1]
        tmp = np.arange(1, param['nl_order_max']+1)
        first_nl_order = np.append(np.concatenate((tmp, tmp[::-1])), tmp[1])[::-1]
        first_nl_order -= 1

        # Inverse Vandermonde matrix for each set with same amplitude
        for idx in range(0, param['K_phase']):
            indexes = np.where(phase_vec == idx)
            tmp_mixing = mixing_mat[:, first_nl_order[idx]::2]
            if nb_term[idx] == param['K_amp']:
                tmp_result = np.dot(np.linalg.inv(tmp_mixing),
                                    out_per_phase[:,idx,:])
            else:
                tmp_result = np.dot(np.linalg.pinv(tmp_mixing),
                                    out_per_phase[:,idx,:])
            if param['output'] == 'orders':
                term_combinatoric[indexes, :] = tmp_result
            elif param['output'] in ['terms', 'real_terms']:
                binomial_factor = np.diag(1/binomial(n_vec[indexes],
                                                     q_vec[indexes]))
                term_combinatoric[indexes, :] = \
                                        np.dot(binomial_factor, tmp_result)

        # Function output
        # (either the nonlinear homogeneous orders of the output of the real
        # signal, or the complex or real terms obtained using binomial
        # decomposition)
        if param['output'] == 'orders':
            orders = np.zeros((param['nl_order_max'], len_sig))
            start = 0
            for ind_n in range(param['nl_order_max']):
                end = start + ind_n + 2
                orders[ind_n, :] = np.real_if_close(np.sum( \
                                          term_combinatoric[start:end], axis=0))
                start += ind_n + 2
            return orders
        elif param['output'] == 'terms':
            if param['out_type'] == 'array':
                return term_combinatoric
            elif param['out_type'] == 'dict':
                term_combinatoric_dict = dict()
                for n in range(1, param['nl_order_max']+1):
                    for q in range(0, n+1):
                        ind = ind = (n*(n+1))//2 + q - 1
                        term_combinatoric_dict[(n, q)] = term_combinatoric[ind]
                return term_combinatoric_dict
        elif param['output'] == 'real_terms':
            nb_real_terms = (param['nb_term'] + param['nl_order_max']//2) // 2
            real_terms = np.zeros((nb_real_terms, len_sig))
            real_terms_dict = dict()
            for n in range(1, param['nl_order_max']+1):
                for q in range(0, 1+n//2):
                    ind = (n*(n-1))//2 + q
                    ind1 = (n*(n+1))//2 + q - 1
                    ind2 = (n*(n+1))//2 + (n - q) - 1
                    if ind1 == ind2:
                        real_terms[ind] = np.real_if_close( \
                                              term_combinatoric[ind1])
                    else:
                        real_terms[ind] = np.real_if_close( \
                                              term_combinatoric[ind1] + \
                                              term_combinatoric[ind2])
                    real_terms_dict[(n, q)] = real_terms[ind]
            if param['out_type'] == 'array':
                return real_terms
            elif param['out_type'] == 'dict':
                return real_terms_dict


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    from pyvi.simulation.systems import second_order_w_nl_damping
    import matplotlib.pyplot as plt

    system = second_order_w_nl_damping(gain=1, f0=100,
                                       damping=0.2, nl_coeff=[1e-1, 3e-5])
    fs = 4410
    T = 0.2
    time_vector = np.arange(0, T, 1/fs)
    f1 = 100
    f2 = 133
    input_sig_cplx = np.exp(2j * np.pi * f1 * time_vector) + \
                     np.exp(2j * np.pi * f2 * time_vector)
    input_sig_real = np.real(input_sig_cplx)

    data_cplx, param_cplx = simu_collection(input_sig_cplx, system, fs=fs,
                                            hold_opt=1, name='test_cplx',
                                            method='complex',
                                            param={'nl_order_max' :3,
                                                   'rho': 2})

    data_real, param_real = simu_collection(input_sig_real, system, fs=fs,
                                            hold_opt=1, name='test_real',
                                            method='boyd',
                                            param={'nl_order_max' :3,
                                                   'negative_gain': False,
                                                   'gain': 2})

    data_phase, param_phase = simu_collection(input_sig_cplx, system, fs=fs,
                                              hold_opt=1, name='test_phase+amp',
                                              method='phase+amp',
                                              param={'nl_order_max' :3,
                                                     'gain': 0.5,
                                                     'output': 'terms'})

    out_order_est_cplx = order_separation(data_cplx['output_collection'],
                                          param_cplx['sep_method'],
                                          param_cplx['sep_param'])
    order_max_cplx = data_cplx['output_by_order'].shape[0]
    out_order_est_real = order_separation(data_real['output_collection'],
                                          param_real['sep_method'],
                                          param_real['sep_param'])
    order_max_real = data_real['output_by_order'].shape[0]
    out_order_est_phase = order_separation(data_phase['output_collection'],
                                           param_phase['sep_method'],
                                           param_phase['sep_param'])
    order_max_phase = data_phase['output_by_order'].shape[0]
    param_phase['sep_param']['output'] = 'orders'
    out_order_est_phase_2 = order_separation(data_phase['output_collection'],
                                             param_phase['sep_method'],
                                             param_phase['sep_param'])

    plt.figure('Method complex - True and estimated orders')
    plt.clf()
    for n in range(order_max_cplx):
        plt.subplot(order_max_cplx, 2, 2*n+1)
        plt.plot(data_cplx['time'],
                 np.real(data_cplx['output_by_order'][n]), 'b')
        plt.subplot(order_max_cplx, 2, 2*n+2)
        plt.plot(data_cplx['time'], np.real(out_order_est_cplx[n]), 'r')
    plt.show()

    plt.figure('Method real - True and estimated orders')
    plt.clf()
    for n in range(order_max_real):
        plt.subplot(order_max_real, 2, 2*n+1)
        plt.plot(data_real['time'], data_real['output_by_order'][n], 'b')
        plt.subplot(order_max_real, 2, 2*n+2)
        plt.plot(data_real['time'], out_order_est_real[n], 'r')
    plt.show()

    plt.figure('Method phase + amplitude - Estimated terms')
    plt.clf()
    N = order_max_phase
    nb_col = 2*(N+1)
    shape = (order_max_phase, nb_col)
    for n in range(1, order_max_phase+1):
        for q in range(0,n+1):
            odd = n%2
            ax = plt.subplot2grid(shape, (n-1, N - n + 2*q), colspan=2)
            ind = (n*(n+1))//2 + q - 1
            ax.plot(data_phase['time'], np.real(out_order_est_phase[ind]), 'b')
            ax.plot(data_phase['time'], np.imag(out_order_est_phase[ind]), 'r')

    plt.figure('Method phase + amplitude - Estimated terms (FFT)')
    plt.clf()
    nfft = 2**int(np.log2(out_order_est_phase.shape[-1])+1)
    f_vec = np.fft.fftfreq(nfft, 1/fs)
    spectrum = fftpack.fft(out_order_est_phase, n=nfft)
    for n in range(1, order_max_phase+1):
        for q in range(0,n+1):
            odd = n%2
            ax = plt.subplot2grid(shape, (n-1, N - n + 2*q), colspan=2)
            ind = (n*(n+1))//2 + q - 1
            ax.plot(f_vec, np.abs(spectrum[ind]))

    plt.figure('Method phase + amplitude - True and estimated orders')
    plt.clf()
    for n in range(order_max_phase):
        plt.subplot(order_max_phase, 2, 2*n+1)
        plt.plot(data_phase['time'], data_phase['output_by_order'][n], 'b')
        plt.subplot(order_max_phase, 2, 2*n+2)
        plt.plot(data_phase['time'], out_order_est_phase_2[n], 'r')

    plt.show()