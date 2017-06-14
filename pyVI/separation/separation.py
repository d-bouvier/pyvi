# -*- coding: utf-8 -*-
"""
Toolbox for nonlinear order separation.

This package creates classes for nonlinear homogeneous order separation of
Volterra series.

Class
-----
SeparationMethod :
    Base class for order separation methods.
AS :
    Class for Amplitude-based Separation method.
PS :
    Class for Phase-based Separation method.
PAS :
    Class for Phase-and-Amplitude-based Separation method.

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
from scipy.fftpack import ifft
from ..utilities.mathbox import binomial


#==============================================================================
# Class
#==============================================================================

#TODO add condition number

class SeparationMethod:
    """
    Base class for order separation methods.

    Parameters
    ----------
    N : int
        Number of orders to separate (truncation order of the Volterra series)
    K : int
        Number of tests signals needed for the method
    factors : (K, 1) array_like
        Factors applied to the base signal in order to create the test signals

    Attributes
    ----------
    N, K, factors

    Methods
    -------
    gen_inputs(signal)
        Creates and returns the collection of input test signals
    process_output(output_coll)
        Process output signals and returns the separated order
    """

    def __init__(self, N, K, factors):
        self.N = N
        self.K = K
        self.factors = factors

    def gen_inputs(self, signal):
        """
        Creates and returns the collection of input test signals.

        Parameters
        ----------
        signal : array_like
            Input signal.

        Returns
        -------
        input_coll : (K, ...)
            Collection of the K input test signals.
        """

        return np.tensordot(self.factors, signal, axes=0)

    def process_outputs(self, output_coll):
        """
        Process outputs signals and returns the separated order.

        Parameters
        ----------
        output_coll : (K, ...) array_like
            Collection of the K output signals.
        """
        pass


class AS(SeparationMethod):
    """
    """
    #TODO docstring

    def __init__(self, N=3, gain=1.51, negative_gain=True, K=None):
        """
        """
        #TODO docstring

        nb_test = N if K is None else K
        self.gain = gain
        self.negative_gain = negative_gain
        SeparationMethod.__init__(self, N, nb_test,
                                  self._gen_amp_factors(nb_test))

    def _gen_amp_factors(self, nb_test):
        """
        """
        #TODO docstring

        tmp_vec = np.arange(nb_test)
        return (-1)**(tmp_vec*self.negative_gain) * \
                self.gain**(tmp_vec // (1+self.negative_gain))

    def process_outputs(self, output_coll):
        """
        """
        #TODO docstring

        mixing_mat = np.vander(self.factors, N=self.N+1, increasing=True)[:,1::]
        return self._inverse_vandermonde_mat(output_coll, mixing_mat)

    @staticmethod
    def _inverse_vandermonde_mat(output_coll, mixing_mat):
        """
        """
        #TODO docstring

        is_square = mixing_mat.shape[0] == mixing_mat.shape[1]

        if is_square: # Square matrix
            return np.dot(np.linalg.inv(mixing_mat), output_coll)
        else: # Non-square matrix (pseudo-inverse)
            return np.dot(np.linalg.pinv(mixing_mat), output_coll)


class PS(SeparationMethod):
    """
    """
    #TODO docstring

    def __init__(self, N=3, rho=1):
        """
        """
        #TODO docstring

        self.rho = rho
        SeparationMethod.__init__(self, N, N, self._gen_phase_factors(N))

    def _gen_phase_factors(self, nb_test):
        """
        """
        #TODO docstring
        self.w = np.exp(- 1j * 2 * np.pi / nb_test)
        return self.rho * self.w**np.arange(nb_test)

    def process_outputs(self, output_coll):
        """
        """
        #TODO docstring

        estimation = self._inverse_fft(output_coll, self.N)
        if self.rho == 1:
            return np.roll(estimation, -1, axis=0)
        else:
            demixing_vec = np.vander([1/self.rho], N=self.N, increasing=True)
            return demixing_vec.T * np.roll(estimation, -1, axis=0)

    @staticmethod
    def _inverse_fft(output_coll, N):
        """
        """
        #TODO docstring

        return ifft(output_coll, n=N, axis=0)


class PAS(PS, AS):
    """
    """
    #TODO docstring

    negative_gain = False
    rho = 1

    def __init__(self, N=3, gain=1.51):
        """
        """
        #TODO docstring

        self.gain = gain
        self.K_amp = (N + 1) // 2
        self.amp_vec = AS._gen_amp_factors(self, self.K_amp)

        self.K_phase = 2*N + 1
        phase_vec = PS._gen_phase_factors(self, self.K_phase)

        nb_test = self.K_phase * self.K_amp
        self.nb_term = (N * (N + 3)) // 2
        factors = np.tensordot(self.amp_vec, phase_vec, axes=0).flatten()
        SeparationMethod.__init__(self, N, nb_test, factors)

    def gen_inputs(self, signal):
        """
        """
        #TODO docstring

        return 2 * np.real(SeparationMethod.gen_inputs(self, signal))

    def process_outputs(self, output_coll, raw_mode=False):
        """
        """
        #TODO docstring

        sig_shape = output_coll.shape[1:]

        out_per_phase = np.zeros((self.K_amp, self.K_phase) + sig_shape,
                                 dtype='complex128')
        if raw_mode:
            combinatorial_terms = dict()
        else:
            out_by_order = np.zeros((self.N,) + sig_shape, dtype='complex128')

        mixing_mat = np.vander(self.amp_vec, N=self.N+1, increasing=True)[:,1::]

        # Inverse DFT for each set with same amplitude
        for idx in range(self.K_amp):
            start = idx * self.K_phase
            end = start + self.K_phase
            out_per_phase[idx] = PS._inverse_fft(output_coll[start:end],
                                                 self.K_phase)

        # Computation of indexes and necessary vector
        tmp = np.arange(1, self.N+1)
        first_nl_order = np.concatenate((tmp[1:2], tmp, tmp[::-1]))

        # Inverse Vandermonde matrix for each set with same phase
        for phase_idx in range(self.K_phase):
            col_idx = np.arange(first_nl_order[phase_idx], self.N+1, 2) - 1
            tmp = AS._inverse_vandermonde_mat(out_per_phase[:, phase_idx],
                                              mixing_mat[:, col_idx])
            if raw_mode:
                for ind in range(tmp.shape[0]):
                    n = first_nl_order[phase_idx] + 2*ind
                    q = ((n - phase_idx) % self.K_phase) // 2
                    combinatorial_terms[(n, q)] = tmp[ind] / binomial(n, q)
            else:
                for ind in range(tmp.shape[0]):
                    n = first_nl_order[phase_idx] + 2*ind
                    out_by_order[n-1] += tmp[ind]

        # Function output
        if raw_mode:
            return combinatorial_terms
        else:
            return np.real_if_close(out_by_order)