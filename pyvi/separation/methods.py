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

Last modified on 19 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import warnings as warn
import numpy as np
import scipy.fftpack as sc_fft
from ..utilities.mathbox import binomial


#==============================================================================
# Class
#==============================================================================

#TODO add condition number

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
            np.vander(self.factors, N=self.N+1, increasing=True)[:,1::]
        return self._inverse_mixing_mat(output_coll, mixing_mat)

    @staticmethod
    def _inverse_mixing_mat(output_coll, mixing_mat):
        """
        Resolves the vandermonde system via inverse or pseudo-inverse.
        """

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

    @staticmethod
    def _inverse_fft(output_coll, N):
        """
        Invert Discrete Fourier Transform using the FFT algorithm.
        """

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
        return _PS._inverse_fft(output_coll, self.nb_phase)


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
        output_by_order = np.zeros((self.N,) + sig_shape, dtype='complex128')
        if raw_mode:
            combinatorial_terms = dict()

        mixing_mat = \
            np.vander(self.amp_vec, N=self.N+1, increasing=True)[:,1::]

        # Inverse DFT for each set with same amplitude
        for idx in range(self.nb_amp):
            start = idx * self.nb_phase
            end = start + self.nb_phase
            out_per_phase[idx] = PS._inverse_fft(output_coll[start:end],
                                                 self.nb_phase)

        # Computation of indexes and necessary vector
        tmp = np.arange(1, self.N+1)
        first_nl_order = np.concatenate((tmp[1:2], tmp, tmp[::-1]))

        # Inverse Vandermonde matrix for each set with same phase
        for phase_idx in range(self.nb_phase):
            col_idx = np.arange(first_nl_order[phase_idx], self.N+1, 2) - 1
            tmp = AS._inverse_mixing_mat(out_per_phase[:, phase_idx],
                                         mixing_mat[:, col_idx])

            for ind in range(tmp.shape[0]):
                n = first_nl_order[phase_idx] + 2*ind
                output_by_order[n-1] += tmp[ind]
                if raw_mode:
                    q = ((n - phase_idx) % self.nb_phase) // 2
                    combinatorial_terms[(n, q)] = tmp[ind] / binomial(n, q)

        # Checking that estimated orders are real signals
        output_by_order = np.real_if_close(output_by_order)
        if np.iscomplexobj(output_by_order):
            output_by_order = np.real(output_by_order)
            message = '\nEstimated orders have non-negligible imaginary ' + \
                      'parts. Only real parts have been returned, but ' + \
                      'this indicates a probable malfunction in PAS method.\n'
            warn.showwarning(message, UserWarning, __file__, 458, line='')

        # Function output
        if raw_mode:
            return output_by_order, combinatorial_terms
        else:
            return output_by_order


class PAS_v2(PAS):
    """
    Class for Phase-based Separation method using fewer signals.

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
    nb_amp : int
        Number of different amplitudes used.
    nb_phase_vec : list(int)
        List of the number of different phases used.
    nb_term : int
        Total number of combinatorial terms.
    amp_vec : array_like (of length nb_amp)
        Vector regrouping all amplitudes used.
    nb_inv : int
        Number of matrix inversion required to do the full separation.
    mat : 2D-array_like (of size (K, K))
        Mixing matrix.
    ind : dict((int, int): int)
        Index where the corresponding (p, q) term is stored.
    list_nq : dict(int: (int, int))
        List of 'p, q) terms appearing in a given matrix inversion.
    ind_coll : dict(int: (int, int))
         Index of all output test signals appearing in a given matrix
         inversion.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_output(output_coll, raw_mode=False)
        Process outputs and returns estimated orders or combinatorial terms.

    See also
    --------
    PAS: Parent class
    _SeparationMethod, PS, AS
    """

    def __init__(self, N=3, gain=1.51):

        self.gain = gain
        self.nb_term = (N * (N + 3)) // 2

        self.nb_amp = (N + 1) // 2
        self.amp_vec = self._gen_amp_factors(self.nb_amp)

        self.nb_phase_vec = (2*N + 1) - 4*np.arange(self.nb_amp)
        self.nb_inv = self.nb_phase_vec[-1]

        factors = []
        for ii, nb_phase in enumerate(self.nb_phase_vec):
            factors.append(self.amp_vec[ii] * \
                           self._gen_phase_factors(nb_phase))

        _SeparationMethod.__init__(self, N, np.sum(self.nb_phase_vec),
                                   np.concatenate(factors))

        self.mat = np.zeros((self.K, self.K))
        self.ind = dict()
        self.list_nq = dict([(n, []) for n in range(self.nb_inv)])
        self.ind_coll = dict([(n, set()) for n in range(self.nb_inv)])
        offsets = np.cumsum(self.nb_phase_vec) - self.nb_phase_vec

        for n in range(1, self.N+1):
            for q in range(n+1):
                if q == n/2:
                    idx = q-1
                elif q > n/2:
                    idx = n-q
                else:
                    idx = q
                phase = (n - 2*q)
                ind_test = phase % self.nb_phase_vec[-1]
                ind_coll = offsets + phase % self.nb_phase_vec
                self.ind[(n, q)] = offsets[idx] + \
                                   phase % self.nb_phase_vec[idx]
                self.mat[ind_coll, self.ind[(n, q)]] = self.amp_vec**n
                self.list_nq[ind_test].append((n, q))
                self.ind_coll[ind_test].update(set(ind_coll))

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
        out_per_phase = np.zeros((self.K,)+sig_shape, dtype='complex128')
        output_by_order = np.zeros((self.N,) + sig_shape, dtype='complex128')
        if raw_mode:
            combinatorial_terms = dict()

        # Inverse DFT for each set with same amplitude
        start = 0
        for nb_phase in self.nb_phase_vec:
            end = start + nb_phase
            out_per_phase[start:end] = PS._inverse_fft(output_coll[start:end],
                                                       nb_phase)
            start += nb_phase

        # Inverse Vandermonde matrix for each set with same phase
        for ind_test in range(self.nb_inv):
            idx = sorted(self.ind_coll[ind_test])
            idx2 = list(idx)
            if (self.N % 2) and ((self.K - 3) in idx):
                idx2.remove(self.K - 3)
            tmp = AS._inverse_mixing_mat(out_per_phase[idx],
                                         self.mat[idx, :][:, idx2])

            for (n, q) in self.list_nq[ind_test]:
                ind = np.where(idx == self.ind[(n, q)])[0][0]
                output_by_order[n-1] += tmp[ind]
                if raw_mode:
                    combinatorial_terms[(n, q)] = tmp[ind] / binomial(n, q)

        # Checking that estimated orders are real signals
        output_by_order = np.real_if_close(output_by_order)
        if np.iscomplexobj(output_by_order):
            output_by_order = np.real(output_by_order)
            message = '\nEstimated orders have non-negligible imaginary ' + \
                      'parts. Only real parts have been returned, but this' + \
                      ' indicates a probable malfunction in PAS_v2 method.\n'
            warn.showwarning(message, UserWarning, __file__, 619, line='')

        # Function output
        if raw_mode:
            return output_by_order, combinatorial_terms
        else:
            return output_by_order