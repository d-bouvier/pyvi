# -*- coding: utf-8 -*-
"""
Module for numerical simulation of system given a state-space representation.

Function
--------
simulation :
    Compute the simulation of a nonlinear system for a given input.

Class
-----
SimulationObject :
    Class for simulation of a system given by a NumericalStateSpace object.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import itertools as itr
import numpy as np
import scipy as sc
import scipy.signal as sc_sig
from .combinatorics import make_pq_combinatorics
from ..system.statespace import NumericalStateSpace
from ..utilities.mathbox import array_symmetrization


#==============================================================================
# Class
#==============================================================================

class SimulationObject:
    """
    Class for simulation of a system given by a NumericalStateSpace object.

    Parameters
    ----------
    system : NumericalStateSpace
        System to simulate, represented by its NumericalStateSpace object.
    fs : int, optional, (default=44100)
        Sampling frequency of input and output signals .
    nl_order_max : int, optional (default=3)
        Order truncation of the simulation.
    holder_order : int, optional (default=1)
        Order of sample holder applied to emulate continuous time dynamics.
    resampling : boolean, optional (default=False)
        If true, signals are upsampled before simulation, downsampled after.

    Attributes
    ----------
    fs_orig : int
        Sampling frequency of input and output signals.
    nl_order_max : int
    holder_order : int
    resampling : boolean
    fs : int
        Sampling frequency used in the simulation.
    mpq_combinatoric : dict(int: list(tuple(int, int, int, int)))
        See make_pq_combinatorics().
    npq_combinatoric : dict(int: list(tuple(int, int, int, int)))
        See make_pq_combinatorics().
    holder_bias_mat : dict(int: numpy.ndarray)
        Matrices representing the bias due to the sample holder.
    dim : dict(str: int)
        See NumericalStateSpace.
    A_m, B_m, C_m, D_m : array-like (numpy.ndarray)
        See NumericalStateSpace.
    mpq, npq : dict((int, int): tensor-like (numpy.ndarray))
        See NumericalStateSpace.

    Methods
    -------
    simulation(input_signal, out_opt='output')
        Computes the numerical simulation for a given input signal.
    compute_kernels(T, which='both')
        Computes the kernels of the system for a given memory length.
    _compute_time_kernels(T)
        Computes the time kernels for a given memory length.
    _compute_freq_kernels(T)
        Computes the frequency kernels for a given memory length.

    See also
    --------
    NumericalStateSpace : object for state-space representation a system.
    make_pq_combinatorics : combinatorics of pq-functions used in a system.
    """

    def __init__(self, system: NumericalStateSpace, fs=44100, nl_order_max=3,
                 holder_order=1, resampling=False):

        # Initialize simulation options
        self.fs_orig = fs
        self.nl_order_max = nl_order_max
        self.holder_order = holder_order
        self.resampling = resampling
        if self.resampling:
            self.fs = self.nl_order_max * self.fs_orig
        else:
            self.fs = self.fs_orig

        # List of Mpq combinations
        self.mpq_combinatoric = make_pq_combinatorics(system.mpq,
                                                      self.nl_order_max,
                                                      system.pq_symmetry)

        # List of Mpq combinations
        self.npq_combinatoric = make_pq_combinatorics(system.npq,
                                                      self.nl_order_max,
                                                      system.pq_symmetry)

        # Filter matrix and holder bias matrices
        self.A_inv = sc.linalg.inv(system.A_m)

        self.filter_mat_orig = self._compute_filter(self.fs_orig, system.A_m)
        self.holder_bias_mat_orig = self._compute_holder(self.fs_orig,
                                                         self.filter_mat_orig,
                                                         system.dim['state'])
        if self.resampling:
            self.filter_mat = self._compute_filter(self.fs, system.A_m)
            self.holder_bias_mat = self._compute_holder(self.fs,
                                                        self.filter_mat,
                                                        system.dim['state'])
        else:
            self.filter_mat = self.filter_mat_orig
            self.holder_bias_mat = self.holder_bias_mat_orig

        # Copy system dimensions, matrices and pq-functions
        self.dim = system.dim.copy()

        self.A_m = system.A_m.copy()
        self.B_m = system.B_m.copy()
        self.C_m = system.C_m.copy()
        self.D_m = system.D_m.copy()

        self.mpq = system.mpq.copy()
        self.npq = system.npq.copy()


    def _compute_filter(self, fs, A_m):
        """
        Compute the discrete filter matrix for a given sampling frequency.
        """

        return sc.linalg.expm(A_m / fs)


    def _compute_holder(self, fs, filter_mat, dim_state):
        """
        Compute the holder bias matrices for a given sampling frequency.
        """

        temp_mat_1 = filter_mat - np.identity(dim_state)
        temp_mat_2 = np.dot(self.A_inv, temp_mat_1)
        holder_bias_mat = dict()
        holder_bias_mat[0] = temp_mat_2.copy()
        if self.holder_order == 1:
            holder_bias_mat[1] = np.dot(self.A_inv, filter_mat) - \
                                 fs * np.dot(self.A_inv, temp_mat_2)
        return holder_bias_mat


    #=============================================#

    def simulation(self, input_signal, out_opt='output'):
        """
        Computes the numerical simulation for a given input signal.

        Parameters
        ----------
        input_signal : numpy.ndarray
            Input signal.
        out_opt : {'output', 'output_by_order', 'state', 'state_by_order', \
                   'all'}, optional (default='output')
            Option that defines what the function returns.

        Returns
        -------
        output_signal : numpy.ndarray
            Output of the system.
        output_by_order : numpy.ndarray
            Output of the system, separated in each order of nonlinearity.
        state_signal : numpy.ndarray
            States of the system.
        state_by_order : numpy.ndarray
            States of the system, separated in each order of nonlinearity.

        In function of the ``out`` option, this function returns:
            - ``output_signal`` (if ``out`` == 'output')
            - ``output_by_order`` (if ``out`` == 'output_by_order')
            - ``state_signal`` (if ``out`` == 'state')
            - ``state_by_order`` (if ``out`` == 'state_by_order')
            - ``output_by_order``, ``state_by_order`` (if ``out`` == 'all')
        """

        ####################
        ## Initialization ##
        ####################

        input_sig = input_signal.copy()
        dtype = input_sig.dtype

        # Enforce good shape when input dimension is 1
        if self.dim['input'] == 1:
            self.B_m.shape = (self.dim['state'], self.dim['input'])
            self.D_m.shape = (self.dim['output'], self.dim['input'])
            input_sig.shape = (self.dim['input'], input_sig.shape[0])

        # Upsampling (if wanted)
        if self.resampling:
            input_sig = sc_sig.resample_poly(input_sig, self.nl_order_max,
                                             1, axis=1)
        sig_len = input_sig.shape[1]

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

        # Correction of bias due to ADC converter (with holder of order 0 or 1)
        if self.holder_order == 0:
            bias_1sample_lag = self.holder_bias_mat[0]

            def holder_bias(mpq_output, n):
                state_by_order[n-1, :, 1::] += np.dot(bias_1sample_lag,
                                                      mpq_output[:, 0:-1])
        elif self.holder_order == 1:
            bias_0sample_lag = self.holder_bias_mat[0] - \
                               self.holder_bias_mat[1]
            bias_1sample_lag = self.holder_bias_mat[1]

            def holder_bias(mpq_output, n):
                state_by_order[n-1, :, :] += np.dot(bias_0sample_lag,
                                                    mpq_output)
                state_by_order[n-1, :, 1::] += np.dot(bias_1sample_lag,
                                                      mpq_output[:, 0:-1])

        # Filter function (simply a matrix product by 'filter_mat')
        def filtering(n):
            for k in np.arange(sig_len-1):
                state_by_order[n-1, :, k+1] += \
                    np.dot(self.filter_mat, state_by_order[n-1, :, k])

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
                mpq_output = \
                    nb * pq_computation(p, q, [m-1 for m in order_set],
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
                output_by_order[n-1, :, :] += nb * \
                    pq_computation(p, q, [m-1 for m in order_set],
                                   self.npq[(p, q)])

        ######################
        ## Function outputs ##
        ######################

        # Downsampling(if necessary)
        if self.resampling:
            state_by_order = sc_sig.resample_poly(state_by_order, 1,
                                                  self.nl_order_max,
                                                  axis=2)
            output_by_order = sc_sig.resample_poly(output_by_order, 1,
                                                   self.nl_order_max,
                                                   axis=2)

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

    #=============================================#

    def compute_kernels(self, T, which='both'):
        """
        Computes the kernels of the system for a given memory length.

        Parameters
        ----------
        T : float
            Memory length in time.
        which : {'time', 'freq', both'}, optional (default='both')
            Option to choose the output that the function returns.

        Returns
        -------
        volterra_kernels : dict(int: numpy.ndarray)
            Time kernels.
        transfer_kernels : dict(int: numpy.ndarray)
            Frequency kernels.

        In function of the ``which`` option, this function returns:
            - ``volterra_kernels`` (if ``which`` == 'time')
            - ``transfer_kernels`` (if ``which`` == 'freq')
            - ``volterra_kernels``, ``transfer_kernels`` (if ``which`` == \
            'both')
        """

        if which == 'time':
            return self._compute_time_kernels(T)

        elif which == 'both':
            volterra_kernels = self._compute_time_kernels(T)
            transfer_kernels = dict()
            for n, kernel in volterra_kernels.items():
                transfer_kernels[n] = np.fft.fftshift(sc.fftpack.fftn(kernel))

            return volterra_kernels, transfer_kernels

        elif which == 'freq':
            return self._compute_freq_kernels(T)

    def _compute_time_kernels(self, T):
        """
        Computes the time kernels for a given memory length.

        Parameters
        ----------
        T : float
            Memory length in time.

        Returns
        -------
        kernels_in2out : dict(int: numpy.ndarray)
            Time kernels.
        """

        ####################
        ## Initialization ##
        ####################

        # Input-to-state and input-to-output kernels initialization
        len_ker = 1 + int(self.fs_orig * T)
        kernels_in2state = dict()
        kernels_in2out = dict()
        shape = (self.dim['state'],)
        for n in range(1, self.nl_order_max+1):
            shape += (len_ker,)
            kernels_in2state[n] = np.zeros(shape)

        # Dirac-delta
        dirac = np.zeros((self.dim['input'], len_ker))
        dirac[:, 0] = 1

        # Enforce good shape when input dimension is 1
        if self.dim['input'] == 1:
            self.B_m.shape = (self.dim['state'], self.dim['input'])
            self.D_m.shape = (self.dim['output'], self.dim['input'])
        if self.dim['output'] == 1:
            self.C_m.shape = (self.dim['output'], self.dim['state'])
            self.D_m.shape = (self.dim['output'], self.dim['input'])

        ##################################################
        ## Creation of functions for kernel computation ##
        ##################################################

        # Computation of the Mpq/Npq functions (given as tensors)
        def pq_computation(n, p, q, order_set, pq_tensor):
            temp_arg = ()
            min_ind = p + q
            for count in range(p):
                max_ind = min_ind + order_set[count]
                temp_arg += (kernels_in2state[order_set[count]],)
                temp_arg += ([count] + list(range(min_ind, max_ind)),)
                min_ind = max_ind
            for count in range(q):
                temp_arg += (dirac, [p+count, min_ind+count])
            temp_arg += (list(range(p + q + n)),)
            return np.tensordot(pq_tensor, np.einsum(*temp_arg), p+q)

        # Correction of bias due to ADC converter (with holder of order 0 or 1)
        if self.holder_order == 0:
            bias_1sample_lag = self.holder_bias_mat_orig[0]

            def holder_bias(mpq_output, n):
                idx_in = [slice(None)] + [slice(len_ker-1)] * n
                idx_out = [slice(None)] + [slice(1, len_ker)] * n
                kernels_in2state[n][idx_out] += \
                    np.tensordot(bias_1sample_lag, mpq_output[idx_in], 1)
        elif self.holder_order == 1:
            bias_0sample_lag = self.holder_bias_mat_orig[0] - \
                               self.holder_bias_mat_orig[1]
            bias_1sample_lag = self.holder_bias_mat_orig[1]

            def holder_bias(mpq_output, n):
                kernels_in2state[n] += np.tensordot(bias_0sample_lag,
                                                    mpq_output, 1)
                idx_in = [slice(None)] + [slice(len_ker-1)] * n
                idx_out = [slice(None)] + [slice(1, len_ker)] * n
                kernels_in2state[n][idx_out] += \
                    np.tensordot(bias_1sample_lag, mpq_output[idx_in], 1)

        # Filter function
        def filtering(n):
            for ind in range(n, n*(len_ker-1)+1):
                for indexes in itr.filterfalse(lambda x: sum(x)-ind,
                                               itr.product(range(1, len_ker),
                                                           repeat=n)):
                    idx_in = [slice(None)] + [(m-1) for m in indexes]
                    idx_out = [slice(None)] + list(indexes)
                    kernels_in2state[n][idx_out] += \
                            np.tensordot(self.filter_mat_orig,
                                         kernels_in2state[n][idx_in], 1)

        ########################
        ## Kernel computation ##
        ########################

        ## Dynamical equation ##
        # Linear term
        holder_bias(self.B_m.dot(dirac), 1)
        filtering(1)
        # Nonlinear terms (due to Mpq functions)
        for n, elt in sorted(self.mpq_combinatoric.items()):
            for p, q, order_set, nb in elt:
                mpq_output = nb * pq_computation(n, p, q, order_set,
                                                 self.mpq[(p, q)])
                holder_bias(mpq_output, n)
            for ind in range(self.dim['state']):
                kernels_in2state[n][ind] = \
                    array_symmetrization(kernels_in2state[n][ind])
            filtering(n)

        ## Output equation ##
        # Linear and nonlinear terms due to matrix C
        for n in range(1, self.nl_order_max+1):
            kernels_in2out[n] = np.tensordot(self.C_m, kernels_in2state[n], 1)
        # Linear term due to matrix D
        kernels_in2out[1] += self.D_m.dot(dirac)
        # Other nonlinear terms (due to Npq functions)
        for n, elt in sorted(self.npq_combinatoric.items()):
            for p, q, order_set, nb in elt:
                kernels_in2out[n] += nb * pq_computation(n, p, q, order_set,
                                                         self.npq[(p, q)])
            for ind in range(self.dim['output']):
                kernels_in2out[n][ind] = \
                    array_symmetrization(kernels_in2out[n][ind])

        ######################
        ## Function outputs ##
        ######################

        if (self.dim['input'] == 1) or (self.dim['output'] == 1):
            for n in range(1, self.nl_order_max+1):
                kernels_in2state[n] = np.squeeze(kernels_in2state[n])
                kernels_in2out[n] = np.squeeze(kernels_in2out[n])

        return kernels_in2out

    def _compute_freq_kernels(self, T):
        """
        Computes the frequency kernels for a given memory length.

        Parameters
        ----------
        T : float
            Memory length in time.

        Returns
        -------
        kernels_in2out : dict(int: numpy.ndarray)
            Frequency kernels.
        """

        ####################
        ## Initialization ##
        ####################

        # Frequency vector
        len_ker = 1 + int(self.fs_orig * T)
        freq_vec = np.fft.fftshift(np.fft.fftfreq(len_ker, d=1/self.fs_orig))

        # Input-to-state and input-to-output kernels initialization
        kernels_in2state = dict()
        kernels_in2out = dict()
        shape = (self.dim['state'],)
        for n in range(1, self.nl_order_max+1):
            shape += (len_ker,)
            kernels_in2state[n] = np.zeros(shape, dtype=np.complex)

        # Enforce good shape when input dimension is 1
        if self.dim['input'] == 1:
            self.B_m.shape = (self.dim['state'], self.dim['input'])
            self.D_m.shape = (self.dim['output'], self.dim['input'])
        if self.dim['output'] == 1:
            self.C_m.shape = (self.dim['output'], self.dim['state'])
            self.D_m.shape = (self.dim['output'], self.dim['input'])

        # Dirac-delta in frequency domain, with bias due to the sampler holder
        if self.holder_order == 0:
            input_freq = np.exp(- 1j*np.pi/self.fs_orig) * \
                         np.sinc(freq_vec/self.fs_orig)
        elif self.holder_order == 1:
            input_freq = np.sinc(freq_vec/self.fs_orig)**2
        input_freq = np.reshape(input_freq, (self.dim['input'], len_ker))

        ##################################################
        ## Creation of functions for kernel computation ##
        ##################################################

        # Computation of the Mpq/Npq functions (given as tensors)
        def pq_computation(n, p, q, order_set, pq_tensor):
            temp_arg = ()
            min_ind = p + q
            for count in range(p):
                max_ind = min_ind + order_set[count]
                temp_arg += (kernels_in2state[order_set[count]],)
                temp_arg += ([count] + list(range(min_ind, max_ind)),)
                min_ind = max_ind
            for count in range(q):
                temp_arg += (input_freq, [p+count, min_ind+count])
            temp_arg += (list(range(p + q + n)),)
            return np.tensordot(pq_tensor, np.einsum(*temp_arg), p+q)

        # Filter function
        def filtering(n):
            freq_tensor = np.zeros((len_ker,)*n)
            freq_tensor += freq_vec
            idx = [slice(None)]
            for ind in range(1, n):
                idx += [None]
                freq_tensor += freq_vec[idx]
            s = np.reshape(2j*np.pi*freq_tensor, freq_tensor.shape + (1, 1))
            identity = np.identity(self.dim['state'])
            filter_values = np.linalg.inv(s * identity - self.A_m)
            return np.einsum('...ij,j...->i...', filter_values,
                             kernels_in2state[n])
            return np.tensordot(filter_values, kernels_in2state[n], 1)

        ########################
        ## Kernel computation ##
        ########################

        ## Dynamical equation ##
        # Linear term
        kernels_in2state[1] = np.dot(self.B_m, input_freq)
        kernels_in2state[1] = filtering(1)
        # Nonlinear terms (due to Mpq functions)
        for n, elt in sorted(self.mpq_combinatoric.items()):
            for p, q, order_set, nb in elt:
                mpq_output = nb * pq_computation(n, p, q, order_set,
                                                 self.mpq[(p, q)])
                kernels_in2state[n] += mpq_output
            for ind in range(self.dim['state']):
                kernels_in2state[n][ind] = \
                            array_symmetrization(kernels_in2state[n][ind])
            kernels_in2state[n] = filtering(n)

        ## Output equation ##
        # Linear and nonlinear terms due to matrix C
        for n in range(1, self.nl_order_max+1):
            kernels_in2out[n] = np.tensordot(self.C_m, kernels_in2state[n], 1)
        # Linear term due to matrix D
        kernels_in2out[1] += self.D_m.dot(input_freq)
        # Other nonlinear terms (due to Npq functions)
        for n, elt in sorted(self.npq_combinatoric.items()):
            for p, q, order_set, nb in elt:
                kernels_in2out[n] += nb * pq_computation(n, p, q, order_set,
                                                         self.npq[(p, q)])
            for ind in range(self.dim['output']):
                kernels_in2out[n][ind] = \
                    array_symmetrization(kernels_in2out[n][ind])

        ######################
        ## Function outputs ##
        ######################

        if (self.dim['input'] == 1) or (self.dim['output'] == 1):
            for n in range(1, self.nl_order_max+1):
                kernels_in2state[n] = np.squeeze(kernels_in2state[n])
                kernels_in2out[n] = np.squeeze(kernels_in2out[n])

        return kernels_in2out
