# -*- coding: utf-8 -*-
"""
Module for Volterra series nonlinear order separation.

This package creates classes for several nonlinear homogeneous order
separation methods.

Class
-----
_SeparationMethod :
    Asbstract base class for order separation methods.
AS :
    Class for Amplitude-based Separation method.
CPS :
    Class for Complex Phase-based Separation method using complex signals.
HPS :
    Class for Phase-based Separation method into homophase signals.
_AbstractPS :
    Abstract base class for Phase-based Separation method using real signals.
PS :
    Class for Phase-based Separation method using real signals (and 2D-DFT).
PAS :
    Class for Phase-and-Amplitude-based Separation method using real signals.

Functions
---------
    Creates the Vandermonde matrix due to the nonlinear orders homogeneity.

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
from ..utilities.decorators import inherit_docstring


#==============================================================================
# Class
#==============================================================================

class _SeparationMethod:
    """
    Asbstract base class for order separation methods.

    Parameters
    ----------
    N : int
        Number of orders to separate (truncation order of the Volterra series).
    factors : array_like
        Factors applied to the base signal in order to create the test signals.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    factors : array_like
        Vector of length K regrouping all factors.
    condition_numbers : list(float)
        List of condition numbers of all matrix inverted during separation.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_output(output_coll)
        Process outputs and returns estimated orders.
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
            ``signal``); first dimension of ``input_coll`` is of length K.
        """

        return np.tensordot(self.factors, signal, axes=0)

    def process_outputs(self, output_coll):
        """
        Process outputs and returns estimated orders.

        Parameters
        ----------
        output_coll : numpy.ndarray
            Collection of the K output signals; first dimension should be of
            length K.

        Returns
        -------
        output_by_order : numpy.ndarray
            Estimation of the nonlinear homogeneous orders; first dimension of
            ``output_by_order`` is of length N.
        """

        self.condition_numbers = []


class AS(_SeparationMethod):
    """
    Class for Amplitude-based Separation method.

    Parameters
    ----------
    N : int
        Number of orders to separate (truncation order of the Volterra series).
    gain : float, optional (default=0.64)
        Gain factor in amplitude between consecutive test signals.
    negative_gain : boolean, optional (default=True)
        Defines if amplitudes with negative values can be used; this greatly
        improves separation.
    K : int, optional (default=None)
        Number of tests signals needed for the method; must be greater than or
        equal to N; if None, will be set equal to N.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    nb_amp : int
        Number of amplitude factors; equal to K for AS.
    factors : array_like
        Vector of length K regrouping all factors.
    gain : float
        Amplitude factor between consecutive test signals.
    negative_gain : boolean
        Boolean for use of negative values amplitudes.
    mixing_mat : numpy.ndarray
        Mixing matrix between orders and output.
    condition_numbers : list(float)
        List of condition numbers of all matrix inverted during separation.

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
        """Generates the vector of amplitude factors."""

        tmp_vec = np.arange(self.nb_amp)
        return (-1)**(tmp_vec*self.negative_gain) * \
                self.gain**(tmp_vec // (1+self.negative_gain))

    @inherit_docstring
    def process_outputs(self, output_coll):
        _SeparationMethod.process_outputs(self, None)
        return self._solve(output_coll, self.mixing_mat)

    def _solve(self, sig_coll, mixing_mat):
        """Solve the linear system via inverse or pseudo-inverse."""

        self.condition_numbers.append(np.linalg.cond(mixing_mat))
        is_square = mixing_mat.shape[0] == mixing_mat.shape[1]
        if is_square:
            return np.dot(np.linalg.inv(mixing_mat), sig_coll)
        else:
            return np.dot(np.linalg.pinv(mixing_mat), sig_coll)


class CPS(_SeparationMethod):
    """
    Class for Complex Phase-based Separation method using complex signals.

    Parameters
    ----------
    N : int
        Number of orders to separate (truncation order of the Volterra series).
    nb_phase : int
        Number of phase factors used; should be greater than N; choosing N
        large leads to a more robust method but also to more test signals.
    rho : float, optional (default=1.)
        Rejection factor value for dealing with the order aliasing effect;
        must be less than 1 to reject higher-orders; must be close to 1 to
        not enhance noise measurement.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    nb_phase : int
        Number of phase factors; equal to K for CPS and HPS.
    factors : array_like
        Vector of length K regrouping all factors.
    rho : float
        Rejection factor.
    w : float
        Initial phase factor.
    fft_axis : int, class attribute
        Axis along which to compute the inverse FFT.
    condition_numbers : list(float)
        List of condition numbers of all matrix inverted during separation.

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
        """Generates the vector of dephasing factors."""

        self.w = np.exp(- 1j * 2 * np.pi / self.nb_phase)
        vec = np.arange(self.nb_phase)/self.nb_phase
        return self.rho * np.exp(- 2j * np.pi * vec)

    @inherit_docstring
    def process_outputs(self, output_coll):
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
        """Inverse Discrete Fourier Transform using the FFT algorithm."""

        self.condition_numbers.append(1)
        return sc_fft.ifft(output_coll, n=self.nb_phase, axis=self.fft_axis)


class HPS(CPS):
    """
    Class for Phase-based Separation method into homophase signals.

    Parameters
    ----------
    N : int
        Number of nonlinear orders (truncation order of the Volterra series).
    nb_phase : int
        Number of phase factors used; should be greater than N; choosing N
        large leads to a more robust method but also to more test signals.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    nb_phase : int
        Number of phase factors; equal to K for CPS and HPS.
    factors : array_like
        Vector of length K regrouping all factors.
    rho : float, class attribute
        Rejection factor; equal to 1 (not used) for HPS, PS and PAS.
    w : float
        Initial phase factor.
    fft_axis : int or tuple(int), class attribute
        Axis along which inverse FFT is computed; equal to 0 for HPS and PAS.
    condition_numbers : list(float)
        List of condition numbers of all matrix inverted during separation.

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
            ``signal``);  first dimension of ``input_coll`` is of length K.
        signal_cplx : numpy.ndarray
            Complex version of ``signal`` obtained using Hilbert transform;
            only returned if ``signal`` is not complex-valued
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
        output_coll : numpy.ndarray
            Collection of the K output signals; first dimension should be of
            length K.

        Returns
        -------
        homophase : numpy.ndarray
            Estimation of the homophase signals; phases are along the first
            dimension, in the following order: [0, 1, ... N, -N, ..., -1].
        """

        _SeparationMethod.process_outputs(self, None)

        temp = self._ifft(output_coll)
        return np.concatenate((temp[0:self.N+1], temp[-self.N:]), axis=0)


class _AbstractPS(HPS):
    """
    Abstract base class for Phase-based Separation method using real signals.

    Parameters
    ----------
    N : int
        Number of nonlinear orders (truncation order of the Volterra series).
    nb_phase : int
        Number of phase factors used; should be greater than N; choosing N
        large leads to a more robust method but also to more test signals.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    nb_phase : int
        Number of phase factors.
    factors : array_like
        Vector of length K regrouping all factors.
    rho : float, class attribute
        Rejection factor; equal to 1 (not used) for HPS, PS and PAS.
    w : float
        Initial phase factor.
    fft_axis : int or tuple(int), class attribute
        Axis along which inverse FFT is computed.
    mixing_mat : dict(int: numpy.ndarray)
        Dictionnary of mixing matrix between orders and output for each phase.
    nq_tuples : dict(int: list((int, int)))
        Dictionnary of list oof tuples (n, q) for each phase.
    c2r_mat : numpy.ndarray, class attribute
        Matrix for taking into account conjuguated terms in estimation.
    condition_numbers : list(float)
        List of condition numbers of all matrix inverted during separation.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_output(output_coll, raw_mode=False)
        Process outputs and returns estimated orders or combinatorial terms.

    See also
    --------
    HPS: Parents class
    CPS, _SeparationMethod
    """

    negative_gain = False
    c2r_mat = np.array([[1., 0], [1., 0], [0, 1.], [0, -1.]])

    def _create_necessary_matrix_and_index(self):
        """"Create mixing matrix and list of (n, q) tuple for each phase."""

        raise NotImplementedError

    def process_outputs(self, output_coll, raw_mode=False):
        """
        Process outputs and returns estimated orders or combinatorial terms.

        Parameters
        ----------
        output_coll : numpy.ndarray
            Collection of the K output signals; first dimension should be of
            length K.
        raw_mode : boolean, optional (default=False)
            If True, only returns eestimated orders; else also returns
            estimated combinatorial terms.

        Returns
        -------
        output_by_order : numpy.ndarray
            Estimation of the nonlinear homogeneous orders; first dimension of
            ``output_by_order`` is of length N.
        combinatorial_terms : dict((int, int): numpy.ndarray)
            Dictionnary of the estimated combinatorial terms for each couple
            (n, q) where n is the nonlinear order and q the number of times
            where the conjuguate input signal appears.
        """

        _SeparationMethod.process_outputs(self, None)

        # Initialization
        combinatorial_terms = dict()
        output_by_order = np.zeros((self.N,) + output_coll.shape[1:])

        # Regroup by phase with an inverse DFT
        out_per_phase = self._regroup_per_phase(output_coll)

        # Extract combinatorial terms from homophase signals
        for phase in range(self.N+1):
            sigs = self._corresponding_sigs(out_per_phase, phase)
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
        """Compute homophase signals using DFT algorithm."""

        output_coll_2d = self._from_1d_to_2d(output_coll)
        return self._ifft(output_coll_2d)

    def _from_1d_to_2d(self, coll_1d):
        """"Reshape the collection of signals form 1D to 2D."""

        raise NotImplementedError

    def _corresponding_sigs(self, out_per_phase, phase):
        """"Returns homophase signals, splitting real and imaginary part."""

        raise NotImplementedError


class PS(_AbstractPS):
    """
    Class for Phase-based Separation method using real signals (and 2D-DFT).

    Parameters
    ----------
    N : int
        Number of nonlinear orders (truncation order of the Volterra series).
    nb_phase : int
        Number of phase factors used; should be greater than N; choosing N
        large leads to a more robust method but also to more test signals.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    nb_phase : int
        Number of phase factors.
    factors : array_like
        Vector of length K regrouping all factors.
    rho : float, class attribute
        Rejection factor; equal to 1 (not used) for HPS, PS and PAS.
    w : float
        Initial phase factor.
    fft_axis : int or tuple(int), class attribute
        Axis along which inverse FFT is computed; equal to (0, 1) for PS.
    mixing_mat : dict(int: numpy.ndarray)
        Dictionnary of mixing matrix between orders and output for each phase.
    nq_tuples : dict(int: list((int, int)))
        Dictionnary of list oof tuples (n, q) for each phase.
    c2r_mat : numpy.ndarray, class attribute
        Matrix for taking into account conjuguated terms in estimation.
    condition_numbers : list(float)
        List of condition numbers of all matrix inverted during separation.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_output(output_coll, raw_mode=False)
        Process outputs and returns estimated orders or combinatorial terms.

    See also
    --------
    _AbstractPS: Parents class
    HPS, CPS, _SeparationMethod
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

    @inherit_docstring
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

    @inherit_docstring
    def _from_1d_to_2d(self, coll_1d):
        shape = (self.nb_phase,)*2 + coll_1d.shape[1:]
        coll_2d = np.zeros(shape)

        iter_obj = itr.combinations_with_replacement(range(self.nb_phase), 2)
        for ind, (idx1, idx2) in enumerate(iter_obj):
            coll_2d[idx1, idx2] = coll_1d[ind]
            if idx1 != idx2:
                coll_2d[idx2, idx1] = coll_1d[ind]

        return coll_2d

    @inherit_docstring
    def _ifft(self, output_coll_2d):
        self.condition_numbers.append(1)
        spectrum_2d = sc_fft.ifft2(output_coll_2d, axes=self.fft_axis)
        spectrum_2d = sc_fft.fftshift(spectrum_2d, axes=self.fft_axis)

        diff = self.nb_phase - self._compute_required_nb_phase(self.N)
        if not (diff // 2):
            slice_obj = slice(diff % 2, None)
        else:
            slice_obj = slice(diff % 2 + diff // 2, -(diff // 2))

        return spectrum_2d[slice_obj, slice_obj][::-1, :]

    @inherit_docstring
    def _corresponding_sigs(self, out_per_phase, phase):
        dec_diag = (self.N + 1 - phase) // 2
        slice_obj = slice(dec_diag, -dec_diag) if dec_diag else slice(None)
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
    Class for Phase-and-Amplitude-based Separation method using real signals.

    Parameters
    ----------
    N : int
        Number of nonlinear orders (truncation order of the Volterra series).
    gain : float, optional (default=0.64)
        Gain factor in amplitude between the input test signals.
    nb_phase : int
        Number of phase factors used; should be greater than N; choosing N
        large leads to a more robust method but also to more test signals.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    nb_phase : int
        Number of phase factors.
    nb_amp : int
        Number of amplitude factors.
    factors : array_like
        Vector of length K regrouping all factors.
    rho : float, class attribute
        Rejection factor; equal to 1 (not used) for HPS, PS and PAS.
    w : float
        Initial phase factor.
    gain : float
        Amplitude factor between consecutive test signals.
    negative_gain : boolean, class attribute
        Boolean for use of negative values amplitudes; equal to False for PAS.
    fft_axis : int or tuple(int), class attribute
        Axis along which inverse FFT is computed; equal to 0 for HPS and PAS.
    mixing_mat : dict(int: numpy.ndarray)
        Dictionnary of mixing matrix between orders and output for each phase.
    nq_tuples : dict(int: list((int, int)))
        Dictionnary of list oof tuples (n, q) for each phase.
    c2r_mat : numpy.ndarray, class attribute
        Matrix for taking into account conjuguated terms in estimation.
    condition_numbers : list(float)
        List of condition numbers of all matrix inverted during separation.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_output(output_coll, raw_mode=False)
        Process outputs and returns estimated orders or combinatorial terms.

    See also
    --------
    _AbstractPS, AS: Parents class
    HPS, CPS, _SeparationMethod
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

    @inherit_docstring
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

    @inherit_docstring
    def _from_1d_to_2d(self, coll_1d):
        shape_2d = (self.HPS_obj.nb_phase, self.nb_amp) + coll_1d.shape[1:]
        return coll_1d.reshape(shape_2d)

    @inherit_docstring
    def _ifft(self, output_coll_2d):
        return self.HPS_obj.process_outputs(output_coll_2d)

    @inherit_docstring
    def _corresponding_sigs(self, out_per_phase, phase):
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
    """
    Creates the Vandermonde matrix due to the nonlinear orders homogeneity.

    Parameters
    ----------
    N : int
        Number of orders to separate (truncation order of the Volterra series).
    factors : array_like
        Factors applied to the base signal in order to create the test signals.

    Returns
    -------
    matrix: np.ndarray (of size=(len(factors), N))
        Mixing matrix of the Volterra orders in the output signals.
    """

    return np.vander(factors, N=N+1, increasing=True)[:, 1::]
