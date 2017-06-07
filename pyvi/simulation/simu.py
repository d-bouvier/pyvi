# -*- coding: utf-8 -*-
"""
Module for numerical simulation of system given its state-space representation.

Function
--------
simulation :
    Compute the simulation of a nonlinear system for a given input.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 07 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from scipy import linalg
from .combinatorics import make_pq_combinatorics
from ..system.statespace import NumericalStateSpace

#==============================================================================
# Class
#==============================================================================

class SimulationObject:
    """
    """
    #TODO docstring

    def __init__(self, system: NumericalStateSpace, fs=44100, nl_order_max=3,
                 holder_order=1, resampling=False):
        """
        """
        #TODO docstring

        # Initialize simulation options
        self.fs = fs
        self.nl_order_max = nl_order_max
        self.holder_order = holder_order
        self.resampling = resampling

        # Filter matrix
        sampling_time = 1/self.fs
        self.A_inv = linalg.inv(system.A_m)
        self.filter_mat = linalg.expm(system.A_m * sampling_time)

        # List of Mpq combinations
        self.mpq_combinatoric = make_pq_combinatorics(system.mpq,
                                                      self.nl_order_max,
                                                      system.pq_symmetry)

        # List of Mpq combinations
        self.npq_combinatoric = make_pq_combinatorics(system.npq,
                                                      self.nl_order_max,
                                                      system.pq_symmetry)

        # Holder bias matrices
        temp_mat_1 = self.filter_mat - np.identity(system.dim['state'])
        temp_mat_2 = np.dot(self.A_inv, temp_mat_1)
        self.holder_bias_mat = dict()
        self.holder_bias_mat[0] = temp_mat_2.copy()
        if self.holder_order == 1:
            self.holder_bias_mat[1] = np.dot(self.A_inv, self.filter_mat) - \
                            (1/sampling_time) * np.dot(self.A_inv, temp_mat_2)

        # Copy system dimensions, matrices and pq-functions
        self.dim = system.dim.copy()

        self.A_m = system.A_m.copy()
        self.B_m = system.B_m.copy()
        self.C_m = system.C_m.copy()
        self.D_m = system.D_m.copy()

        self.mpq = system.mpq.copy()
        self.npq = system.npq.copy()

    #=============================================#

    def simulation(self, input_signal, out_opt='output'):
        """
        Compute the simulation of a nonlinear system for a given input.

        Parameters
        ----------
        input_sig : numpy.ndarray
            Input signal.
        system : StateSpace
            Parameters of the system to simulate.
        fs : int, optional
            Sampling frequency.
        nl_order_max : int, optional
            Maximum order of nonlinearity to take into account.
        hold_opt : {0, 1}, optional
            Type of sample-holder of the ADC converter to emulate.
        out : {'output', 'output_by_order', 'all'}, optional
            Option to choose the output.

        Returns
        -------
        output_sig : numpy.ndarray
            Output of the system.
        output_by_order : numpy.ndarray
            Output of the system, separated in each order of nonlinearity.
        state_by_order : numpy.ndarray
            States of the system, separated in each order of nonlinearity.

        In function of the ``out`` option, this function returns:
            - ``output_sig`` (if ``out`` == 'output')
            - ``output_by_order`` (if ``out`` == 'output_by_order')
            - ``state`` (if ``out`` == 'state')
            - ``output_sig``, ``state_by_order``, and ``output_by_order`` (if \
            ``out`` == 'all')

        """
        #TODO update docstring

        ####################
        ## Initialization ##
        ####################

        input_sig = input_signal.copy()
        dtype = input_sig.dtype

        # Enforce good shape when input dimension is 1
        if self.dim['input'] == 1:
            sig_len = input_sig.shape[0]
            self.B_m.shape = (self.dim['state'], self.dim['input'])
            self.D_m.shape = (self.dim['output'], self.dim['input'])
            input_sig.shape = (self.dim['input'], sig_len)
        else:
            sig_len = input_sig.shape[0]

        # By-order state and output initialization
        state_by_order = np.zeros((self.nl_order_max, self.dim['state'],
                                   sig_len), dtype)
        output_by_order = np.zeros((self.nl_order_max, self.dim['output'],
                                    sig_len), dtype)

        ##########################################
        ## Creation of functions for simulation ##
        ##########################################

        # Computation of the Mpq/Npq functions (given as tensors)
        def pq_computation(p, q, order_set, pq_tensor):
            temp_arg = ()
            for count in range(p):
                temp_arg += (state_by_order[order_set[count]],)
                temp_arg += ([count, p+q],)
            for count in range(q):
                temp_arg += (input_sig, [p+count, p+q])
            temp_arg += (list(range(p+q+1)),)
            return np.tensordot(pq_tensor, np.einsum(*temp_arg), p+q)

        # Correction of the bias due to ADC converter (with holder of order 0 or 1)
        if self.holder_order == 0:
            bias_1sample_lag = self.holder_bias_mat[0]
            def holder_bias(mpq_output, n):
                state_by_order[n-1,:,1::] += np.dot(bias_1sample_lag,
                                                    mpq_output[:,0:-1])
        elif self.holder_order == 1:
            bias_0sample_lag = self.holder_bias_mat[0] - self.holder_bias_mat[1]
            bias_1sample_lag = self.holder_bias_mat[1]
            def holder_bias(mpq_output, n):
                state_by_order[n-1,:,:] += np.dot(bias_0sample_lag, mpq_output)
                state_by_order[n-1,:,1::] += np.dot(bias_1sample_lag,
                                                    mpq_output[:,0:-1])

        # Filter function (simply a matrix product by 'filter_mat')
        def filtering(n):
            for k in np.arange(sig_len-1):
                state_by_order[n-1,:,k+1] += np.dot(self.filter_mat,
                                                    state_by_order[n-1,:,k])

        ##########################
        ## Numerical simulation ##
        ##########################

        ## Dynamical equation ##
        # Linear state
        holder_bias(np.dot(self.B_m, input_sig), 1)
        filtering(1)
        # Nonlinear states (due to Mpq functions)
        for n, elt in sorted(self.mpq_combinatoric.items()):
            for p, q, order_set, nb in elt:
                mpq_output = nb * pq_computation(p, q, [m-1 for m in order_set],
                                                 self.mpq[(p, q)])
                holder_bias(mpq_output, n)
            filtering(n)

        ## Output equation ##
        # Output term due to matrix D
        output_by_order[0] += np.dot(self.D_m, input_sig)
        # Output terms due to matrix C
        for n in range(self.nl_order_max):
            output_by_order[n] += np.dot(self.C_m, state_by_order[n])
        # Other nonlinear output terms (due to Npq functions)
        for n, elt in sorted(self.npq_combinatoric.items()):
            for p, q, order_set, nb in elt:
                output_by_order[n-1,:,:] += nb * \
                pq_computation(p, q, [m-1 for m in order_set], self.npq[(p, q)])

        ######################
        ## Function outputs ##
        ######################

        # Reshaping state (if necessary)
        if self.dim['state'] == 1:
            state_by_order = state_by_order[:, 0, :]

        # Reshaping output (if necessary)
        if self.dim['output'] == 1:
            output_by_order = output_by_order[:, 0, :]

        # Saving data
        self._state_by_order = state_by_order.copy()
        self._output_by_order = output_by_order.copy()

        if out_opt == 'output':
            return output_by_order.sum(0)
        if out_opt == 'output_by_order':
            return output_by_order
        if out_opt == 'state':
            return state_by_order.sum(0)
        if out_opt == 'state_by_order':
            return state_by_order
        if out_opt == 'all':
            return output_by_order, state_by_order
