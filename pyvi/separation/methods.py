# -*- coding: utf-8 -*-
"""
Module for nonlinear order separation.

This package creates classes for nonlinear homogeneous order separation of
Volterra series.

Class
-----
_SeparationMethod :
    Base class for order separation methods.
AS :
    Class for Amplitude-based Separation method.
_PS :
    Class for Phase-based Separation method using complex signals.
PS :
    Class for Phase-based Separation method into homo-phase signals.
PAS :
    Class for Phase-and-Amplitude-based Separation method.
PAS_v2 :
    Class for Phase-and-Amplitude-based Separation method using fewer signals.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 24 Nov. 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import warnings
import numpy as np
import scipy.fftpack as sc_fft
from ..utilities.mathbox import binomial


#==============================================================================
# Class
#==============================================================================

class _SeparationMethod:
    """
    Base class for order separation methods.

    Parameters
    ----------
    N : int
        Number of orders to separate (truncation order of the Volterra series).
    K : int
        Number of tests signals needed for the method.
    factors : array_like (with length K)
        Factors applied to the base signal in order to create the test signals.
    condition_numbers : list(float)
        List of condition numbers of all matrix inverted during separation.

    Attributes
    ----------
    N : int
    K : int
    factors : array_like (of length K)

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_output(output_coll)
        Process outputs.
    """

    def __init__(self, N, K, factors):
        self.N = N
        self.K = K
        self.factors = factors
        self.condition_numbers = []

    def gen_inputs(self, signal):
        """
        Returns the collection of input test signals.

        Parameters
        ----------
        signal : array_like
            Input signal.

        Returns
        -------
        input_coll : numpy.ndarray
            Collection of the K input test signals (each with the same shape as
            ``signal``).
        """

        return np.tensordot(self.factors, signal, axes=0)

    def process_outputs(self, output_coll):
        """
        Process outputs.

        Parameters
        ----------
        output_coll : array_like
            Collection of the K output signals.
        """
        pass


class AS(_SeparationMethod):
    """
    Class for Amplitude-based Separation method.

    Parameters
    ----------
    N : int, optional (default=3)
        Number of orders to separate (truncation order of the Volterra series).
    gain : float, optional (default=1.51)
        Gain factor in amplitude between  the input test signals.
    negative_gain : boolean, optional (default=True)
        Defines if amplitudes with negative values can be used.
    K : int, optional (default=None)
        Number of tests signals needed for the method; must be greater than or
        equal to N; if None, will be set equal to N.

    Attributes
    ----------
    N : int
    K : int
    factors : array_like (of length K)
    gain : float
    negative_gain : boolean

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_output(output_coll)
        Process outputs and returns estimated orders.

    See also
    --------
    _SeparationMethod : Parent class.
    """

    def __init__(self, N=3, gain=1.51, negative_gain=True, K=None):

        nb_test = N if K is None else K
        self.gain = gain
        self.negative_gain = negative_gain
        _SeparationMethod.__init__(self, N, nb_test,
                                   self._gen_amp_factors(nb_test))

    def _gen_amp_factors(self, nb_test):
        """
        Generates the vector of amplitude factors.
        """

        tmp_vec = np.arange(nb_test)
        return (-1)**(tmp_vec*self.negative_gain) * \
                self.gain**(tmp_vec // (1+self.negative_gain))

    def process_outputs(self, output_coll):
        """
        Process outputs and returns estimated orders.

        Parameters
        ----------
        output_coll : array_like
            Collection of the K output signals.

        Returns
        -------
        output_by_order : array_like
            Estimation of the N first nonlinear homogeneous orders.
        """

        mixing_mat = \
            np.vander(self.factors, N=self.N+1, increasing=True)[:, 1::]
        return self._inverse_mixing_mat(output_coll, mixing_mat)

    def _inverse_mixing_mat(self, output_coll, mixing_mat):
        """
        Resolves the vandermonde system via inverse or pseudo-inverse.
        """

        self.condition_numbers.append(np.linalg.cond(mixing_mat))
        is_square = mixing_mat.shape[0] == mixing_mat.shape[1]
        if is_square: # Square matrix
            return np.dot(np.linalg.inv(mixing_mat), output_coll)
        else: # Non-square matrix (pseudo-inverse)
            return np.dot(np.linalg.pinv(mixing_mat), output_coll)


class _PS(_SeparationMethod):
    """
    Class for Phase-based Separation method using complex signals.

    Parameters
    ----------
    N : int, optional (default=3)
        Number of orders to separate (truncation order of the Volterra series).
    rho : float, optional (default=1.)
        Rejection factor value for dealing with the order aliasing effect.

    Attributes
    ----------
    N : int
    K : int
    factors : array_like (of length K)
    rho : float
    w : float
        Complex unit-root used as dephasing factor.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_output(output_coll)
        Process outputs and returns estimated orders.

    See also
    --------
    _SeparationMethod: Parent class
    """

    def __init__(self, N=3, rho=1.):
        self.rho = rho
        _SeparationMethod.__init__(self, N, N, self._gen_phase_factors(N))

    def _gen_phase_factors(self, nb_test):
        """
        Generates the vector of dephasing factors.
        """

        self.w = np.exp(- 1j * 2 * np.pi / nb_test)
        return self.rho * self.w**np.arange(nb_test)

    def process_outputs(self, output_coll):
        """
        Process outputs and returns estimated orders.

        Parameters
        ----------
        output_coll : array_like
            Collection of the K output signals.

        Returns
        -------
        output_by_order : array_like
            Estimation of the N first nonlinear homogeneous orders.
        """

        estimation = self._inverse_fft(output_coll, self.N)
        if self.rho == 1:
            return np.roll(estimation, -1, axis=0)
        else:
            demixing_vec = np.vander([1/self.rho], N=self.N, increasing=True)
            return demixing_vec.T * np.roll(estimation, -1, axis=0)

    def _inverse_fft(self, output_coll, N):
        """
        Invert Discrete Fourier Transform using the FFT algorithm.
        """

        self.condition_numbers.append(1)
        return sc_fft.ifft(output_coll, n=N, axis=0)


class PS(_PS):
    """
    Class for Phase-based Separation method into homo-phase signals.

    Parameters
    ----------
    N : int, optional (default=3)
        Number of nonlinear orders (truncation order of the Volterra series).

    Attributes
    ----------
    N : int
    K : int
    factors : array_like (of length K)
    rho : float (class Attribute, always 1)
    w : float
    nb_phase : int
        Number of different phases used.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_output(output_coll)
        Process outputs and returns estimated homo-phase signals.

    See also
    --------
    _PS: Parents class
    _SeparationMethod
    """

    rho = 1

    def __init__(self, N=3):
        self.nb_phase = 2*N + 1
        phase_vec = self._gen_phase_factors(self.nb_phase)

        _SeparationMethod.__init__(self, N, self.nb_phase, phase_vec)

    def gen_inputs(self, signal):
        """
        Returns the collection of input test signals.

        Parameters
        ----------
        signal : array_like
            Input signal.

        Returns
        -------
        input_coll : numpy.ndarray
            Collection of the K input test signals (each with the same shape as
            ``signal``).

        See also
        --------
        _SeparationMethod.gen_inputs
        """

        return 2 * np.real(_SeparationMethod.gen_inputs(self, signal))

    def process_outputs(self, output_coll):
        return _PS._inverse_fft(self, output_coll, self.nb_phase)


class PAS(PS, AS):
    """
    Class for Phase-and-Amplitude-based Separation method.

    Parameters
    ----------
    N : int, optional (default=3)
        Number of orders to separate (truncation order of the Volterra series).
    gain : float, optional (default=1.51)
        Gain factor in amplitude between  the input test signals.

    Attributes
    ----------
    N : int
    K : int
    factors : array_like (of length K)
    gain : float
    negative_gain : boolean (class Attribute, always False)
    rho : float (class Attribute, always 1)
    w : float
    nb_amp : int
        Number of different amplitudes used.
    nb_phase : int
        Number of different phases used.
    nb_term : int
        Total number of combinatorial terms.
    amp_vec : array_like (of length nb_amp)
        Vector regrouping all amplitudes used.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_output(output_coll, raw_mode=False)
        Process outputs and returns estimated orders or combinatorial terms.

    See also
    --------
    PS, AS: Parents class
    _SeparationMethod
    """

    negative_gain = False

    def __init__(self, N=3, gain=1.51):
        self.gain = gain
        self.nb_amp = (N + 1) // 2
        self.amp_vec = self._gen_amp_factors(self.nb_amp)

        self.nb_phase = 2*N + 1
        phase_vec = self._gen_phase_factors(self.nb_phase)

        nb_test = self.nb_phase * self.nb_amp
        self.nb_term = (N * (N + 3)) // 2
        factors = np.tensordot(self.amp_vec, phase_vec, axes=0).flatten()

        _SeparationMethod.__init__(self, N, nb_test, factors)

    def process_outputs(self, output_coll, raw_mode=False):
        """
        Process outputs and returns estimated orders or combinatorial terms.

        Parameters
        ----------
        output_coll : (K, ...) array_like
            Collection of the K output signals.
        raw_mode : boolean, optional (default=False)
            Option that defines what the function returns.

        Returns
        -------
        output_by_order : numpy.ndarray
            Estimation of the N first nonlinear homogeneous orders.
        combinatorial_terms : dict((int, int): array_like)
            Estimation of the N first nonlinear homogeneous orders.

        This function always return ``output_by_order``; it also returns
        ``combinatorial_terms`` if `raw_mode`` optionis set to True.
        """

        sig_shape = output_coll.shape[1:]

        out_per_phase = np.zeros((self.nb_amp, self.nb_phase) + sig_shape,
                                 dtype='complex128')
        output_by_order = np.zeros((self.N,) + sig_shape)
        if raw_mode:
            combinatorial_terms = dict()

        mixing_mat = \
            np.vander(self.amp_vec, N=self.N+1, increasing=True)[:, 1::]

        # Inverse DFT for each set with same amplitude
        for idx in range(self.nb_amp):
            start = idx * self.nb_phase
            end = start + self.nb_phase
            out_per_phase[idx] = PS._inverse_fft(self, output_coll[start:end],
                                                 self.nb_phase)

        # Computation of indexes and necessary vector
        tmp = np.arange(1, self.N+1)
        first_nl_order = np.concatenate((tmp[1:2], tmp, tmp[::-1]))
        conj_mat = np.array([[1., 0], [1., 0], [0, 1.], [0, -1.]])

        # Inverse Vandermonde matrix for each set with same null phase
        col_idx = np.arange(first_nl_order[0], self.N+1, 2) - 1
        tmp = AS._inverse_mixing_mat(self, np.real(out_per_phase[:, 0]),
                                     mixing_mat[:, col_idx])
        for ind in range(tmp.shape[0]):
            n = first_nl_order[0] + 2*ind
            output_by_order[n-1] += tmp[ind]
            if raw_mode:
                q = ((n - 0) % self.nb_phase) // 2
                combinatorial_terms[(n, q)] = tmp[ind] / binomial(n, q)

        # Inverse Vandermonde matrix for each set with same non-null phase
        for phase_idx in range(1, 1 + self.nb_phase // 2):
            col_idx = np.arange(first_nl_order[phase_idx], self.N+1, 2) - 1
            phase_idx_conj = self.nb_phase - phase_idx
            sigs = np.concatenate((np.real(out_per_phase[:, phase_idx]),
                                   np.real(out_per_phase[:, phase_idx_conj]),
                                   np.imag(out_per_phase[:, phase_idx]),
                                   np.imag(out_per_phase[:, phase_idx_conj])))
            mix_mat = np.kron(conj_mat, mixing_mat[:, col_idx])
            tmp = AS._inverse_mixing_mat(self, sigs, mix_mat)
            ind_dec = tmp.shape[0] // 2
            for ind in range(ind_dec):
                n = first_nl_order[phase_idx] + 2*ind
                output_by_order[n-1] += 2 * tmp[ind]
                if raw_mode:
                    q = ((n - phase_idx) % self.nb_phase) // 2
                    combinatorial_terms[(n, q)] = \
                        (tmp[ind] + 1j*tmp[ind+ind_dec]) / binomial(n, q)

        # Function output
        if raw_mode:
            return output_by_order, combinatorial_terms
        else:
            return output_by_order
