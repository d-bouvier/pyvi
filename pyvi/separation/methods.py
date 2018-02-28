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
        self.mixing_mat = create_vandermonde_mixing_mat(self.factors, self.N)

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
        return self._solve(output_coll, self.mixing_mat)

    def _solve(self, output_coll, mixing_mat):
        """
        Resolves the vandermonde system via inverse or pseudo-inverse.
        """

        self.condition_numbers.append(np.linalg.cond(mixing_mat))
        is_square = mixing_mat.shape[0] == mixing_mat.shape[1]
        if is_square:
            return np.dot(np.linalg.inv(mixing_mat), output_coll)
        else:
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

    fft_axis = 0

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

        estimation = np.roll(self._ifft(output_coll), -1, axis=0)[:self.N]
        if self.rho == 1:
            return estimation
        else:
            vec = np.arange(1, self.N+1)
            demixing_vec = (1/self.rho) ** vec
            demixing_vec.shape = (self.N, 1)
            self.condition_numbers.append(np.diag(demixing_vec))
            return demixing_vec * estimation

    def _ifft(self, output_coll):
        """
        Invert Discrete Fourier Transform using the FFT algorithm.
        """

        self.condition_numbers.append(1)
        return sc_fft.ifft(output_coll, n=self.nb_phase, axis=self.fft_axis)


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
            return 2*np.real(super().gen_inputs(signal_cplx)), signal_cplx
        else:
            return 2*np.real(super().gen_inputs(signal))

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

        temp = self._ifft(output_coll)
        return np.concatenate((temp[0:self.N+1], temp[-self.N:]), axis=0)


class _AbstractPS(HPS):
    """
    Abstract Class for phase-based order separation method using real signals.
    """

    negative_gain = False
    c2r_mat = np.array([[1., 0], [1., 0], [0, 1.], [0, -1.]])

    def _create_necessary_matrix_and_index(self):
        raise NotImplementedError

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

        _SeparationMethod.process_outputs(self, None)

        combinatorial_terms = dict()
        output_by_order = np.zeros((self.N,) + output_coll.shape[1:])

        # Regroup by phase with an inverse DFT
        out_per_phase = self._regroup_per_phase(output_coll)

        # Extract combinatorial terms from homophase signals
        for phase in range(self.N+1):
            sigs = self._compute_sigs(out_per_phase, phase)
            tmp = AS._solve(self, sigs, self.mixing_mat[phase])
            for ind, (n, q) in enumerate(self.nq_tuples[phase]):
                if phase:
                    dec = tmp.shape[0] // 2
                    combinatorial_terms[(n, q)] = (tmp[ind] + 1j*tmp[ind+dec])
                    output_by_order[n-1] += 2 * binomial(n, q) * tmp[ind]
                else:
                    combinatorial_terms[(n, q)] = tmp[ind]
                    output_by_order[n-1] += binomial(n, q) * tmp[ind]

        # Function output
        if raw_mode:
            return output_by_order, combinatorial_terms
        else:
            return output_by_order

    def _regroup_per_phase(self, output_coll):
        """
        Invert Discrete Fourier Transform using the FFT algorithm.
        """

        output_coll_2d = self._from_1d_to_2d(output_coll)
        return self._ifft(output_coll_2d)

    def _from_1d_to_2d(self, coll_1d):
        raise NotImplementedError

    def _ifft(self, output_coll_2d):
        raise NotImplementedError

    def _compute_sigs(self, out_per_phase, phase):
        raise NotImplementedError


class PS(_AbstractPS):
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

    fft_axis = (0, 1)

    def __init__(self, N, nb_phase=None):
        super().__init__(N, nb_phase=nb_phase)

        factors = []
        for w1, w2 in itr.combinations_with_replacement(self.factors, 2):
            factors.append(w1 + w2)

        self.factors = factors
        self.K = len(factors)
        self._create_necessary_matrix_and_index()

    def _create_necessary_matrix_and_index(self):
        self.mixing_mat = dict()
        self.nq_tuples = dict()

        for phase in range(self.N+1):
            start = phase if phase else 2
            current_orders = np.arange(start, self.N+1, 2)
            self.nq_tuples[phase] = [(n, (n-phase)//2) for n in current_orders]

            len_diag = self.N + int((phase % 2) == (self.N % 2))
            p1_start = ((phase + (self.N+1) % 2) // 2) - (self.N // 2)
            p1_end = p1_start + len_diag
            p1_vec = np.arange(p1_start, p1_end)
            tmp_mat = np.zeros((len_diag, len(current_orders)))

            # Computation of the combinatorial factor for each term
            for indp, (p1, p2) in enumerate(zip(p1_vec, p1_vec[::-1])):
                for indn, n in enumerate(current_orders):
                    if (abs(p1) + abs(p2)) <= n:
                        tmp_start = max(0, p1)
                        tmp_end = 1 + (n - abs(p2) + p1) // 2
                        k1_vec = np.arange(tmp_start, tmp_end)
                        k2_vec = k1_vec - p1
                        k3_vec = (n + p2 - (k1_vec + k2_vec)) // 2
                        k4_vec = k3_vec - p2

                        for k in zip(k1_vec, k2_vec, k3_vec, k4_vec):
                            tmp_mat[indp, indn] += multinomial(n, k)

            if phase:
                self.mixing_mat[phase] = np.kron(self.c2r_mat, tmp_mat)
            else:
                self.mixing_mat[phase] = tmp_mat

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

    def _ifft(self, output_coll_2d):
        """
        Invert Discrete Fourier Transform using the FFT algorithm.
        """

        self.condition_numbers.append(1)
        spectrum_2d = sc_fft.ifft2(output_coll_2d, axes=self.fft_axis)
        spectrum_2d = sc_fft.fftshift(spectrum_2d, axes=self.fft_axis)
        return self._truncate_spectrum(spectrum_2d)[::-1, :]

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

    def _compute_sigs(self, out_per_phase, phase):
        dec_diag = (self.N + 1 - phase) // 2
        if dec_diag == 0:
            slice_obj = slice(None)
        else:
            slice_obj = slice(dec_diag, -dec_diag)

        # Regroupment of conjuguate terms (upper and lower diagonal)
        if phase:
            upper_diag = np.diagonal(out_per_phase, offset=phase).T
            lower_diag = np.diagonal(out_per_phase, offset=-phase).T
            return np.concatenate((np.real(upper_diag[slice_obj]),
                                   np.real(lower_diag[slice_obj]),
                                   np.imag(upper_diag[slice_obj]),
                                   np.imag(lower_diag[slice_obj])))
        else:
            return np.real((np.diagonal(out_per_phase).T)[slice_obj])


class PAS(_AbstractPS, AS):
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

    def __init__(self, N, gain=0.64, nb_phase=None):
        self.nb_phase = self._compute_required_nb_phase(N)
        self.HPS_obj = HPS(N, nb_phase=nb_phase)

        AS.__init__(self, (N + 1) // 2, gain=gain,
                    negative_gain=self.negative_gain)
        self.amp_vec = self.factors

        self.nb_term = self.nb_amp * self.nb_phase

        factors = np.tensordot(self.HPS_obj.factors, self.amp_vec, axes=0)
        factors = factors.flatten()

        _SeparationMethod.__init__(self, N, factors)
        self._create_necessary_matrix_and_index()

    def _create_necessary_matrix_and_index(self):
        self.mixing_mat = dict()
        self.nq_tuples = dict()
        global_mixing_mat = create_vandermonde_mixing_mat(self.amp_vec, self.N)

        for phase in range(self.N+1):
            start = phase if phase else 2
            current_orders = np.arange(start, self.N+1, 2)
            self.nq_tuples[phase] = [(n, (n-phase)//2) for n in current_orders]

            tmp_mat = global_mixing_mat[:, current_orders-1]
            for ind, (n, q) in enumerate(self.nq_tuples[phase]):
                tmp_mat[:, ind] *= binomial(n, q)

            if phase:
                self.mixing_mat[phase] = np.kron(self.c2r_mat, tmp_mat)
            else:
                self.mixing_mat[phase] = tmp_mat

    def _from_1d_to_2d(self, coll_1d):
        """
        Invert Discrete Fourier Transform using the FFT algorithm.
        """

        shape_2d = (self.HPS_obj.nb_phase, self.nb_amp) + coll_1d.shape[1:]
        return coll_1d.reshape(shape_2d)

    def _ifft(self, output_coll_2d):
        """
        Invert Discrete Fourier Transform using the FFT algorithm.
        """

        return self.HPS_obj.process_outputs(output_coll_2d)

    def _compute_sigs(self, out_per_phase, phase):
        if phase:
            phase_conj = self.nb_phase - phase
            return np.concatenate((np.real(out_per_phase[phase]),
                                   np.real(out_per_phase[phase_conj]),
                                   np.imag(out_per_phase[phase]),
                                   np.imag(out_per_phase[phase_conj])))
        else:
            return np.real(out_per_phase[0])


#==============================================================================
# Functions
#==============================================================================

def create_vandermonde_mixing_mat(factors, N):
    return np.vander(factors, N=N+1, increasing=True)[:, 1::]
