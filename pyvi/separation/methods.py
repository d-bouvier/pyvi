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
CPS :
    Class for Complex Phase-based Separation method (using complex signals).
HPS :
    Class for Phase-based Separation method into homophase signals.
PAS :
    Class for Phase-and-Amplitude-based Separation method.
PAS_v2 :
    Class for Phase-and-Amplitude-based Separation method using fewer signals.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import warnings
import itertools as itr
import numpy as np
import scipy.fftpack as sc_fft
import scipy.signal as sc_sig
from ..utilities.mathbox import binomial, multinomial


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
    factors : array_like (with length K)
        Factors applied to the base signal in order to create the test signals.

    Attributes
    ----------
    N : int
    factors : array_like (of length K)
    K : int
        Number of tests signals needed for the method.
    condition_numbers : list(float)
        List of condition numbers of all matrix inverted during separation.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_output(output_coll)
        Process outputs.
    """

    def __init__(self, N, factors):
        self.N = N
        self.factors = factors
        self.K = len(factors)
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

        self.condition_numbers = []


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

    def __init__(self, N, gain=0.64, negative_gain=True, K=None):
        self.nb_amp = N if K is None else K
        self.gain = gain
        self.negative_gain = negative_gain
        super().__init__(N, self._gen_amp_factors())

    def _gen_amp_factors(self):
        """
        Generates the vector of amplitude factors.
        """

        tmp_vec = np.arange(self.nb_amp)
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

        _SeparationMethod.process_outputs(self, None)

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


class CPS(_SeparationMethod):
    """
    Class for Complex Phase-based Separation method (using complex signals).

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

    def __init__(self, N, nb_phase=None, rho=1.):
        self.rho = rho

        nb_phase_min = self._compute_required_nb_phase(N)
        if nb_phase is not None:
            if nb_phase < nb_phase_min:
                message = "Specified 'nb_phase' parameter is lower than " + \
                          "the minimum needed ({}) .Instead, minimum was used."
                warnings.warn(message.format(nb_phase_min), UserWarning)
                nb_phase = nb_phase_min
            self.nb_phase = nb_phase
        else:
            self.nb_phase = nb_phase_min

        super().__init__(N, self._gen_phase_factors())

    def _compute_required_nb_phase(self, N):
        """Computes the required minium number of phase."""

        return N

    def _gen_phase_factors(self):
        """
        Generates the vector of dephasing factors.
        """

        self.w = np.exp(- 1j * 2 * np.pi / self.nb_phase)
        vec = np.arange(self.nb_phase)/self.nb_phase
        return self.rho * np.exp(- 2j * np.pi * vec)

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

        _SeparationMethod.process_outputs(self, None)

        estimation = np.roll(self._inverse_fft(output_coll), -1, axis=0)
        if self.rho == 1:
            return estimation[:self.N]
        else:
            vec = np.arange(1, self.N+1)
            demixing_vec = (1/self.rho) ** vec
            demixing_vec.shape = (self.N, 1)
            return demixing_vec * estimation[:self.N]

    def _inverse_fft(self, output_coll):
        """
        Invert Discrete Fourier Transform using the FFT algorithm.
        """

        self.condition_numbers.append(1)
        return sc_fft.ifft(output_coll, n=self.nb_phase, axis=0)


class HPS(CPS):
    """
    Class for Phase-based Separation method into homophase signals.

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
        Process outputs and returns estimated homophase signals.

    See also
    --------
    CPS: Parents class
    _SeparationMethod
    """

    rho = 1

    def __init__(self, N, nb_phase=None):
        super().__init__(N, nb_phase=nb_phase, rho=self.rho)

    def _compute_required_nb_phase(self, N):
        """Computes the required minium number of phase."""

        return 2*N + 1

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
        signal_cplx : numpy.ndarray (only if ``signal`` is not complex)
            Complex version of ``signal`` obtained using Hilbert transform.

        See also
        --------
        _SeparationMethod.gen_inputs
        """

        if not np.iscomplexobj(signal):
            signal_cplx = (1/2) * sc_sig.hilbert(signal)
            return 2*np.real(_SeparationMethod.gen_inputs(self, signal_cplx)),\
                signal_cplx
        else:
            return 2*np.real(_SeparationMethod.gen_inputs(self, signal))

    def process_outputs(self, output_coll):
        """
        Process outputs and returns homophase signals.

        Parameters
        ----------
        output_coll : array_like
            Collection of the K output signals.

        Returns
        -------
        homophase : array_like
            Estimation of the homophase signals for phase -N to N.
        """

        _SeparationMethod.process_outputs(self, None)

        temp = self._inverse_fft(output_coll)
        homophase = np.concatenate((temp[0:self.N+1], temp[-self.N:]), axis=0)
        return homophase


class PS(HPS):
    """
    Class for Real Phase-based Separation method (using 2D-Fourier Transform).

    Parameters
    ----------
    N : int
        Number of nonlinear orders (truncation order of the Volterra series).

    Attributes
    ----------
    N : int
    K : int
    factors : array_like (of length K)
    w : float
    rho : float (class Attribute, always 1)
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

    def __init__(self, N, nb_phase=None):
        super().__init__(N, nb_phase=nb_phase)

        factors = []
        for w1, w2 in itr.combinations_with_replacement(self.factors, 2):
            factors.append(w1 + w2)

        self.factors = factors
        self.K = len(factors)

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
        ``combinatorial_terms`` if `raw_mode`` option is set to True.
        """

        _SeparationMethod.process_outputs(self, None)

        output_by_order = np.zeros((self.N,) + output_coll.shape[1:])
        if raw_mode:
            combinatorial_terms = dict()

        # Computation of the inverse 2D-DFT
        out_per_phase = self._inverse_fft(output_coll)

        # Computation of the complex-2-real matrix
        c2r_mat = np.array([[1., 0], [1., 0], [0, 1.], [0, -1.]])

        # Loop on all diagonals of the 2D-spectrum
        for phase_idx in range(self.N+1):

            # Diagonal length and indexes
            len_diag = self.N + int((phase_idx % 2) == (self.N % 2))
            dec_diag = (self.N + 1 - phase_idx) // 2
            if dec_diag == 0:
                slice_obj = slice(None)
            else:
                slice_obj = slice(dec_diag, -dec_diag)

            # Nonlinear orders that are present in this diagonal
            if phase_idx:
                current_nl_orders = np.arange(phase_idx, self.N+1, 2)
            else:
                current_nl_orders = np.arange(2, self.N+1, 2)
            nb_term = len(current_nl_orders)

            # Initialization
            start = ((phase_idx + (self.N+1) % 2) // 2) - (self.N // 2)
            end = start + len_diag
            p1_vec = np.arange(start, end)
            mixing_mat = np.zeros((len_diag, nb_term))

            # Computation of the combinatorial factor for each term
            for indp, (p1, p2) in enumerate(zip(p1_vec, p1_vec[::-1])):
                for indn, n in enumerate(current_nl_orders):
                    if (abs(p1) + abs(p2)) <= n:
                        tmp_start = max(0, p1)
                        tmp_end = 1 + (n - abs(p2) + p1) // 2
                        k1_vec = np.arange(tmp_start, tmp_end)
                        k2_vec = k1_vec - p1
                        k3_vec = (n + p2 - (k1_vec + k2_vec)) // 2
                        k4_vec = k3_vec - p2

                        for k in zip(k1_vec, k2_vec, k3_vec, k4_vec):
                            mixing_mat[indp, indn] += multinomial(n, k)

            # Regroupment of conjuguate terms (upper and lower diagonal)
            if phase_idx:
                upper_diag = np.diagonal(out_per_phase, offset=phase_idx).T
                lower_diag = np.diagonal(out_per_phase, offset=-phase_idx).T
                sigs = np.concatenate((np.real(upper_diag[slice_obj]),
                                       np.real(lower_diag[slice_obj]),
                                       np.imag(upper_diag[slice_obj]),
                                       np.imag(lower_diag[slice_obj])))
                mix_mat = np.kron(c2r_mat, mixing_mat)
            else:
                sigs = np.real((np.diagonal(out_per_phase).T)[slice_obj])
                mix_mat = mixing_mat

            # Computation of the Mpq terms and nonlinear orders
            tmp = AS._inverse_mixing_mat(self, sigs, mix_mat)
            if phase_idx:
                ind_dec = tmp.shape[0] // 2
                for ind in range(ind_dec):
                    n = phase_idx + 2 * ind
                    q = ((n - phase_idx) % self.nb_phase) // 2
                    output_by_order[n-1] += 2 * binomial(n, q) * tmp[ind]
                    if raw_mode:
                        combinatorial_terms[(n, q)] = \
                            tmp[ind] + 1j*tmp[ind+ind_dec]
            else:
                for ind in range(tmp.shape[0]):
                    n = 2 * (ind+1)
                    q = ((n - phase_idx) % self.nb_phase) // 2
                    output_by_order[n-1] += binomial(n, q) * tmp[ind]
                    if raw_mode:
                        combinatorial_terms[(n, q)] = tmp[ind]

        # Function output
        if raw_mode:
            return output_by_order, combinatorial_terms
        else:
            return output_by_order

    def _inverse_fft(self, output_coll):
        """
        Invert Discrete Fourier Transform using the FFT algorithm.
        """

        output_coll_2d = self._from_1d_to_2d(output_coll)

        self.condition_numbers.append(1)
        phase_spectrum_2d = sc_fft.ifft2(output_coll_2d, axes=(0, 1))
        phase_spectrum_2d = sc_fft.fftshift(phase_spectrum_2d, axes=(0, 1))

        homophase_signals_2d = self._truncate_spectrum(phase_spectrum_2d)

        return homophase_signals_2d[::-1, :]

    def _from_1d_to_2d(self, coll_1d):
        """
        Invert Discrete Fourier Transform using the FFT algorithm.
        """

        shape = (self.nb_phase,)*2 + coll_1d.shape[1:]
        coll_2d = np.zeros(shape)

        iter_obj = itr.combinations_with_replacement(range(self.nb_phase), 2)
        for ind, (idx1, idx2) in enumerate(iter_obj):
            coll_2d[idx1, idx2] = coll_1d[ind]
            if idx1 != idx2:
                coll_2d[idx2, idx1] = coll_1d[ind]

        return coll_2d

    def _truncate_spectrum(self, spectrum_2d):
        """
        Truncate a 2D-spectrum to keep only the bins of the first frequencies.
        """

        diff = self.nb_phase - self._compute_required_nb_phase(self.N)
        if not (diff // 2):
            slice_obj = slice(diff % 2, None)
        else:
            slice_obj = slice(diff % 2 + diff // 2, -(diff // 2))

        return spectrum_2d[slice_obj, slice_obj]


class PAS(HPS, AS):
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

    def __init__(self, N, gain=0.64, nb_phase=None):
        AS.__init__(self, (N + 1) // 2, gain=gain,
                    negative_gain=self.negative_gain)
        self.amp_vec = self.factors

        self.nb_phase = self._compute_required_nb_phase(N)
        self.HPS_obj = HPS(N, nb_phase=nb_phase)

        self.nb_term = self.nb_amp * self.nb_phase

        factors = np.tensordot(self.amp_vec, self.HPS_obj.factors, axes=0)
        factors = factors.flatten()

        _SeparationMethod.__init__(self, N, factors)

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
            start = idx * self.HPS_obj.nb_phase
            end = start + self.HPS_obj.nb_phase
            out_per_phase[idx] = \
                self.HPS_obj.process_outputs(output_coll[start:end])

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
