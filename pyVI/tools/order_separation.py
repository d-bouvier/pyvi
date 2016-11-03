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

#==============================================================================
# Functions
#==============================================================================

def separation_measure(signals_ref, signals_est):
    """
    Measure the efficiency of source separation using SDR, SIR and SAR metrics
    defined in:
    [#vincent2006performance] Emmanuel Vincent, Rémi Gribonval, and Cédric
    Févotte, "Performance measurement in blind audio source separation," IEEE
    Trans. on Audio, Speech and Language Processing, 14(4):1462-1469, 2006.
    """

    nb_src = signals_est.shape[0]

    def sig_projection(signal_est):
        """
        Projection of estimated signal on the reference signals.
        """
        A = np.corrcoef(signals_ref, y=signal_est )
        G = A[0:3, 0:3]
        D = A[3, 0:3]
        try:
            C = np.linalg.solve(G, D).reshape(nb_src, order='F')
        except np.linalg.linalg.LinAlgError:
            C = np.linalg.lstsq(G, D)[0].reshape(nb_src, order='F')
        return np.dot(C, signals_ref)

    sdr = []
    sir = []
    sar = []

    for i in range(nb_src):
        interference_err = sig_projection(signals_est[i]) - signals_ref[i]
        artificial_err = - signals_ref[i] - interference_err
        sdr.append(safe_db(np.sum(signals_ref[i]**2),
                           np.sum((interference_err + artificial_err)**2)))
        sir.append(safe_db(np.sum(signals_ref[i]**2),
                           np.sum((interference_err)**2)))
        sar.append(safe_db(np.sum((signals_ref[i] + interference_err)**2),
                           np.sum((artificial_err)**2)))

    return (sdr, sir, sar)


def safe_db(num, den):
    """
    dB computation with verification that the denominator is not zero.
    """
    if den == 0:
        return np.Inf
    return 10 * np.log10(num / den)


def simu_collection(input_sig, coll_factor, system, fs=44100, N=1, hold_opt=1,
                    dtype='float'):
    """
    Make collection of simulation with inouts derived from a based signal.
    """

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

    return output, out_by_order

