# -*- coding: utf-8 -*-
"""
Module for nonlinear homogeneous order separation of Volterra series.

This module creates classes for several nonlinear homogeneous order
separation methods.

Class
-----
AS :
    Class for Amplitude-based Separation method.
CPS :
    Class for Complex Phase-based Separation method using complex signals.
HPS :
    Class for Phase-based Separation method into homophase signals.
RPS :
    Class for Phase-based Separation method using real signals and 2D-DFT.
PAS :
    Class for Phase-and-Amplitude-based Separation method using real signals.

Notes
-----
Developed for Python 3.6
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

__all__ = ['AS', 'CPS', 'HPS', 'RPS', 'PAS']


#==============================================================================
# Importations
#==============================================================================

import warnings
import itertools as itr
import numpy as np
import scipy.fftpack as sc_fft
import scipy.signal as sc_sig
import scipy.optimize as sc_optim
from .tools import (_create_vandermonde_mixing_mat, _compute_condition_number,
                    _demix_coll)
from ..utilities.mathbox import binomial, multinomial
from ..utilities.tools import inherit_docstring


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
    constant_term : boolean, optional (default=False)
        If True, constant term of the Volterra series (i.e. the order 0) is
        also separated; otherwise it is considered null.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    factors : array_like
        Vector of length `K` regrouping all factors.
    constant_term : boolean
        Whether constant term of the Volterra series is separated.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_outputs(output_coll)
        Process outputs and returns estimated orders.
    get_condition_numbers(p=2)
        Return the list of all condition numbers in the separation method.
    """

    def __init__(self, N, factors, constant_term=False):
        self.N = N
        self.factors = factors
        self.K = len(factors)
        self.constant_term = constant_term
        self._N = self.N + int(self.constant_term)

    def gen_inputs(self, signal):
        """
        Returns the collection of input test signals.

        Parameters
        ----------
        signal : numpy.ndarray
            Input signal; must be equivaent to a 1D array.

        Returns
        -------
        input_coll : numpy.ndarray
            Collection of the input test signals; its shape verifies
            ``input_coll.shape == (self.K,) + signal.shape``.
        """

        if len(np.squeeze(signal).shape) != 1:
            raise ValueError('Multi-dimensional input signals are not (yet) '
                             'handled by separation methods.')

        return np.tensordot(self.factors, signal, axes=0)

    def process_outputs(self, output_coll):
        """
        Process outputs and returns estimated orders.

        Parameters
        ----------
        output_coll : numpy.ndarray
            Collection of the output signals; it should verify
            ``output_coll.shape[0] == self.K``.

        Returns
        -------
        output_by_order : numpy.ndarray
            Estimation of the nonlinear homogeneous orders; it verifies
            ``output_by_order.shape[0] == self.N + self.constant_term`` and
            ``output_by_order.shape[1:] == output_coll.shape[1:]``.
        """

        raise NotImplementedError

    def _update_factors(self, new_factors):
        self.factors = new_factors
        self.K = len(self.factors)

    def _check_parameter(self, nb, name):
        nb_min = getattr(self, '_compute_required_' + name)()
        if nb is not None:
            if nb < nb_min:
                message = "Specified `{}` parameter is lower than " + \
                          "the minimum needed ({}). Instead, minimum was used."
                warnings.warn(message.format(name, nb_min), UserWarning)
                nb = nb_min
            setattr(self, name, nb)
        else:
            setattr(self, name, nb_min)

    def get_condition_numbers(self, p=2):
        """
        Return the list of all condition numbers in the separation method.

        Parameters
        ----------
        p : {None, 1, -1, 2, -2, inf, -inf, 'fro'}, optional (default=2)
            Order of the norm
            :ref:`(see np.linalg.norm for more details) <np.linalg.norm>`.

        Returns
        -------
        condition_numbers : list(float)
            List of all condition numbers.
        """

        return []


class AS(_SeparationMethod):
    """
    Class for Amplitude-based Separation method.

    Parameters
    ----------
    N : int
        Number of orders to separate (truncation order of the Volterra series).
    amplitudes : array_like, optional (default=None)
        Amplitude factors applied to the base signal in order to create the
        test signals; if not specified, those values will be computed using
        parameters `gain`, `negative_amp` and `nb_amp` (if they are given),
        or chosen as the Chebyshev nodes.
    gain : float, optional (default=None)
        Gain factor in amplitude between consecutive test signals; if
        `amplitudes` is specified, this value is not used.
    negative_amp : boolean, optional (default=True)
        Defines if amplitudes with negative values can be used; this greatly
        improves separation; if `amplitudes` is specified, this value is not
        used.
    nb_amp : int, optional (default=None)
        Number of different amplitudes; must be greater than or equal to `N`;
        if None, will be set equal to `N`; if `amplitudes` is specified,
        this value is not used.
    constant_term : boolean, optional (default=False)
        If True, constant term of the Volterra series (i.e. the order 0) is
        also separated; otherwise it is considered null.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    nb_amp : int
        Number of amplitude factors; always equal to `K` for AS.
    factors : array_like
        Vector of length `K` regrouping all factors.
    amplitudes : array_like
        Vector of length `nb_amp` regrouping all amplitudes; always equal to
        `factors` for AS.
    mixing_mat : numpy.ndarray
        Mixing matrix between orders and output.
    constant_term : boolean
        Whether constant term of the Volterra series is separated.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_outputs(output_coll)
        Process outputs and returns estimated orders.
    get_condition_numbers(p=2)
        Return the list of all condition numbers in the separation method.

    Class methods
    -------------
    optimum_gain(N, p=2, gain_min=.1, gain_max=.99, gain_init=None, tol=1e-6,
                 **kwargs)
        Search for the gain that minimizes the maximum condition number.

    See also
    --------
    _SeparationMethod : Parent class.
    """

    def __init__(self, N, amplitudes=None, gain=None, negative_amp=True,
                 nb_amp=None, **kwargs):
        super().__init__(N, [], **kwargs)

        if amplitudes is not None:
            nb_amp = len(amplitudes)
            nb_amp_min = self._compute_required_nb_amp()
            if nb_amp < nb_amp_min:
                raise ValueError('Not enough values in `amplitudes` ' +
                                 '(got {}, '.format(nb_amp) +
                                 'expected at least {})'.format(nb_amp_min))
            self.nb_amp = nb_amp
            self.amplitudes = amplitudes
        else:
            self._check_parameter(nb_amp, 'nb_amp')
            if gain is None and negative_amp is False:
                self.amplitudes = \
                    np.cos(np.pi * np.arange(self.nb_amp)/(2*self.nb_amp-1))
            elif gain is None and negative_amp is True:
                if self.nb_amp % 2:
                    _tmp = np.cos(np.pi * np.arange(self.nb_amp+1)/self.nb_amp)
                    self.amplitudes = np.concatenate((_tmp[:self.nb_amp//2+1],
                                                      _tmp[2+self.nb_amp//2:]))
                else:
                    self.amplitudes = \
                        np.cos(np.pi * np.arange(self.nb_amp)/(self.nb_amp-1))
            else:
                self.amplitudes = self._gen_amplitudes(gain, negative_amp)
        self._update_factors(self.amplitudes)

        self.mixing_mat = \
            _create_vandermonde_mixing_mat(self.factors, self.N,
                                           first_column=self.constant_term)

    def _compute_required_nb_amp(self):
        """Computes the required minium number of amplitude."""

        return self._N

    def _gen_amplitudes(self, gain, negative_amp):
        """Generates the vector of amplitude."""

        tmp_vec = np.arange(self.nb_amp)
        return (-1)**(tmp_vec*negative_amp) * \
                gain**(tmp_vec // (1+negative_amp))

    @inherit_docstring
    def process_outputs(self, output_coll):
        return _demix_coll(output_coll, self.mixing_mat)

    @inherit_docstring
    def get_condition_numbers(self, p=2):
        return [_compute_condition_number(self.mixing_mat, p=p)]

    @classmethod
    def optimum_gain(cls, N, p=2, gain_min=0., gain_max=1., gain_init=None,
                     tol=1e-6, **kwargs):
        """
        Search for the gain that minimizes the condition number.

        Parameters
        ----------
        N : int
            Truncation order.
        p : {None, 1, -1, 2, -2, inf, -inf, 'fro'}, optional (default=2)
            Order of the norm
            :ref:`(see np.linalg.norm for more details) <np.linalg.norm>`.
        gain_min : float, optional (default=0.1)
            Minimum possible value for the gain.
        gain_max : float, optional (default=0.99)
            Maximum possible value for the gain.
        gain_init : float, optional (default=None)
            Starting value of the gian for the optimization procedure; if None,
            it is set as ``(gain_min+gain_max)/2``.
        **kwargs : Keywords arguments passed to the separation method.

        Returns
        -------
        gain : float
            Gain minimizing the condition number computed using norm `p`.
        """

        def func(gain):
            method_obj = cls(N, gain=gain, **kwargs)
            try:
                return np.log10(max(method_obj.get_condition_numbers(p=p)))
            except (ValueError, np.linalg.linalg.LinAlgError):
                return np.inf

        if gain_init is None:
            _nb_amp = kwargs.get('nb_amp', N)
            gain_init = min(max(np.cos(np.pi/(2*_nb_amp)), gain_min), gain_max)

        not_succeed = True
        eps = tol
        while not_succeed:
            results = sc_optim.minimize(func, gain_init, method='TNC',
                                        bounds=[(gain_min, gain_max)],
                                        options={'ftol': tol, 'eps': eps})
            if results.success and results.x[0] != 0 and results.x[0] != 1:
                not_succeed = False
            else:
                eps *= 1e-1

        return results.x[0]


class CPS(_SeparationMethod):
    """
    Class for Complex Phase-based Separation method using complex signals.

    Parameters
    ----------
    N : int
        Number of orders to separate (truncation order of the Volterra series).
    nb_phase : int
        Number of phase factors used; should be greater than `N`; choosing
        `nb_phase` large leads to a more robust method but also to more test
        signals.
    rho : float, optional (default=None)
        Rejection factor value for dealing with the order aliasing effect;
        must be less than 1 to reject higher-orders; must be close to 1 to
        not enhance noise measurement; not used if None.
    constant_term : boolean, optional (default=False)
        If True, constant term of the Volterra series (i.e. the order 0) is
        also separated; otherwise it is considered null.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    nb_phase : int
        Number of phase factors; equal to `K` for CPS.
    factors : array_like
        Vector of length `K` regrouping all factors.
    rho : float
        Rejection factor.
    w : float
        Initial phase factor.
    fft_axis : int, class attribute
        Axis along which to compute the inverse FFT; equal to 0 for CPS.
    constant_term : boolean
        Whether constant term of the Volterra series is separated.

    Methods
    -------
    gen_inputs(signal)
        Returns the collection of input test signals.
    process_outputs(output_coll)
        Process outputs and returns estimated orders.
    get_condition_numbers(p=2)
        Return the list of all condition numbers in the separation method.

    See also
    --------
    _SeparationMethod: Parent class
    """

    fft_axis = 0

    def __init__(self, N, nb_phase=None, rho=None, **kwargs):
        super().__init__(N, [], **kwargs)

        self._check_parameter(nb_phase, 'nb_phase')
        self.rho = rho
        self._update_factors(self._gen_phase_factors())

        power_min = int(not self.constant_term)
        if self.rho is not None:
            self.contrast_vector = (1/self.rho) ** np.arange(power_min,
                                                             self.N+1)

    def _compute_required_nb_phase(self):
        """Computes the required minium number of phase."""

        return self._N

    def _gen_phase_factors(self):
        """Generates the vector of dephasing factors."""

        self.w = np.exp(- 1j * 2 * np.pi / self.nb_phase)
        vec = np.arange(self.nb_phase)/self.nb_phase
        factors = np.exp(- 2j * np.pi * vec)
        if self.rho is not None:
            return self.rho * factors
        else:
            return factors

    @inherit_docstring
    def process_outputs(self, output_coll):
        estimation = self._ifft(output_coll)
        if not self.constant_term:
            estimation = np.roll(estimation, -1, axis=0)
        estimation = estimation[:self._N]
        if self.rho is not None:
            indexes = (slice(None),) + (np.newaxis,) * (estimation.ndim - 1)
            return self.contrast_vector[indexes] * estimation
        else:
            return estimation

    def _ifft(self, output_coll):
        """Inverse Discrete Fourier Transform using the FFT algorithm."""

        return sc_fft.ifft(output_coll, n=self.nb_phase, axis=self.fft_axis)

    @inherit_docstring
    def get_condition_numbers(self, p=2):
        condition_numbers = [self._fft_mat_condition_numbers(p)]
        if self.rho is not None:
            condition_numbers.append(self._contrast_condition_numbers(p))
        return condition_numbers

    def _fft_mat_condition_numbers(self, p):
        if p in [None, 'fro', 2, -2]:
            return 1
        if p in [np.inf, -np.inf, 1, -1]:
            return self.nb_phase

    def _contrast_condition_numbers(self, p):
        return _compute_condition_number(np.diag(self.contrast_vector), p=p)


class HPS(CPS):
    """
    Class for Phase-based Separation method into homophase signals.

    Parameters
    ----------
    N : int
        Number of nonlinear orders (truncation order of the Volterra series).
    nb_phase : int
        Number of phase factors used; should be greater than ``2*N+1``;
        choosing `nb_phase` large leads to a more robust method but also to
        more test signals.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    nb_phase : int
        Number of phase factors; equal to `K` for HPS.
    factors : array_like
        Vector of length `K` regrouping all factors.
    rho : float, class attribute
        Rejection factor; equal to None for HPS.
    w : float
        Initial phase factor.
    fft_axis : int or tuple(int), class attribute
        Axis along which inverse FFT is computed; equal to 0 for HPS.
    constant_term : False
        Whether constant term of the Volterra series is separated; have no
        effect for HPS.

    Methods
    -------
    gen_inputs(signal, return_cplx_sig=False)
        Returns the collection of input test signals.
    process_outputs(output_coll)
        Process outputs and returns estimated homophase signals.
    get_condition_numbers(p=2)
        Return the list of all condition numbers in the separation method.

    See also
    --------
    CPS: Parents class
    _SeparationMethod
    """

    rho = None

    def __init__(self, N, nb_phase=None, **kwargs):
        super().__init__(N, nb_phase=nb_phase, rho=self.rho, **kwargs)

    @inherit_docstring
    def _compute_required_nb_phase(self):
        return 2*self.N + 1

    def gen_inputs(self, signal, return_cplx_sig=False):
        """
        Returns the collection of input test signals.

        Parameters
        ----------
        signal : numpy.ndarray
            Input signal; must be equivaent to a 1D array.
        return_cplx_sig : boolean, optional (default=False)
            If `signal`is real-valued, chosses Whether to return the complex
            signal constructed from its hilbert transform.

        Returns
        -------
        input_coll : numpy.ndarray
            Collection of the input test signals; its shape verifies
            ``input_coll.shape == (self.K,) + signal.shape``.
        signal_cplx : numpy.ndarray
            Complex version of `signal` obtained using Hilbert transform;
            only returned if `signal` is real-valued and `return_cplx_sig` is
            True.
        """

        is_complex = np.iscomplexobj(signal)
        if not is_complex:
            signal_cplx = sc_sig.hilbert(signal)
        else:
            signal_cplx = signal

        input_coll = np.real(super().gen_inputs(signal_cplx))
        if not is_complex and return_cplx_sig:
            return input_coll, signal_cplx
        else:
            return input_coll

    def process_outputs(self, output_coll):
        """
        Process outputs and returns homophase signals.

        Parameters
        ----------
        output_coll : numpy.ndarray
            Collection of the output signals; it should verify
            ``output_coll.shape[0] == self.K``.

        Returns
        -------
        homophase : numpy.ndarray
            Estimation of the homophase signals; it verifies
            ``homophase.shape == (2*self.N+1,) + output_coll.shape[1:]``.
        """

        temp = self._ifft(output_coll)
        return np.concatenate((temp[0:self.N+1], temp[-self.N:]), axis=0)

    @inherit_docstring
    def get_condition_numbers(self, p=2):
        return [self._fft_mat_condition_numbers(p)]


class _AbstractPS(HPS):
    """
    Abstract base class for Phase-based Separation method using real signals.

    Parameters
    ----------
    N : int
        Number of nonlinear orders (truncation order of the Volterra series).
    nb_phase : int
        Number of phase factors used; should be greater than ``2*N+1``;
        choosing `nb_phase` large leads to a more robust method but also to
        more test signals.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    nb_phase : int
        Number of phase factors.
    factors : array_like
        Vector of length `K` regrouping all factors.
    rho : float, class attribute
        Rejection factor; equal to None for _AbstractPS.
    w : float
        Initial phase factor.
    fft_axis : int or tuple(int), class attribute
        Axis along which inverse FFT is computed.
    mixing_mat_dict : dict(int: numpy.ndarray)
        Dictionnary of mixing matrix between orders and output for each phase.
    nq_dict : dict(int: list((int, int)))
        Dictionnary of list of tuples (n, q) for each phase.
    constant_term : False
        Whether constant term of the Volterra series is separated; have no
        effect for HPS.

    Methods
    -------
    gen_inputs(signal, return_cplx_sig=False)
        Returns the collection of input test signals.
    process_outputs(output_coll, raw_mode=False)
        Process outputs and returns estimated orders or interconjugate terms.
    get_condition_numbers(p=2)
        Return the list of all condition numbers in the separation method.

    See also
    --------
    HPS: Parents class
    CPS, _SeparationMethod
    """

    _cplx2real_mat = np.array([[1., 0], [1., 0], [0, 1.], [0, -1.]])

    def _create_nq_dict(self):
        """Create dictionnary of (n, q) tuple for each phase."""

        self.nq_dict = dict()
        self._dec = int(not self.constant_term)
        for phase in range(self.N+1):
            if self.constant_term:
                start = phase
            else:
                start = phase if phase else 2
            current_orders = np.arange(start, self.N+1, 2)
            self.nq_dict[phase] = [(n, (n-phase)//2) for n in current_orders]

    def _create_mixing_matrix_dict(self):
        """"Create dictionnary of mixing matrix for each phase."""

        self.mixing_mat_dict = dict()
        for phase in range(self.N+1):
            tmp_mixing_mat = self._create_tmp_mixing_matrix(phase)
            if phase:
                tmp_mixing_mat = np.kron(self._cplx2real_mat, tmp_mixing_mat)
            self.mixing_mat_dict[phase] = tmp_mixing_mat

    def _create_tmp_mixing_matrix(self, phase):
        """"Create mixing matrix for a given phase."""

        raise NotImplementedError

    def process_outputs(self, output_coll, raw_mode=False):
        """
        Process outputs and returns estimated orders or interconjugate terms.

        Parameters
        ----------
        output_coll : numpy.ndarray
            Collection of the output signals; it should verify
            ``output_coll.shape[0] == self.K``
        raw_mode : boolean, optional (default=False)
            If False, only returns estimated orders; else also returns
            estimated interconjugate terms.

        Returns
        -------
        output_by_order : numpy.ndarray
            Estimation of the nonlinear homogeneous orders; it verifies
            ``output_by_order.shape[0] == self.N + self.constant_term`` and
            ``output_by_order.shape[1:] == output_coll.shape[1:]``.
        interconjugate_terms : dict((int, int): numpy.ndarray)
            Dictionary of the estimated interconjugate terms; contains
            all keys ``(n, q)`` for ``n in range(1, N+1)`` and
            ``q in range(1+n//2)``; each corresponding value verifies
            ``interconjugate_terms[(n, q)].shape == output_coll.shape[1:]``.
        """

        # Initialization
        interconjugate_terms = dict()
        output_by_order = np.zeros((self._N,) + output_coll.shape[1:])

        # Regroup by phase with an inverse DFT
        out_per_phase = self._regroup_per_phase(output_coll)

        # Extract interconjugate terms from homophase signals
        for phase in range(self.N+1):
            sigs = self._corresponding_sigs(out_per_phase, phase)
            tmp = _demix_coll(sigs, self.mixing_mat_dict[phase])
            for ind, (n, q) in enumerate(self.nq_dict[phase]):
                if phase:
                    dec = tmp.shape[0] // 2
                    interconjugate_terms[(n, q)] = (2**n) * \
                                                   (tmp[ind] + 1j*tmp[ind+dec])
                    output_by_order[n-self._dec] += 2*binomial(n, q) * tmp[ind]
                else:
                    interconjugate_terms[(n, q)] = (2**n) * tmp[ind]
                    output_by_order[n-self._dec] += binomial(n, q) * tmp[ind]

        # Function output
        if raw_mode:
            return output_by_order, interconjugate_terms
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

    @inherit_docstring
    def get_condition_numbers(self, p=2):
        condition_numbers = super().get_condition_numbers(p=p)
        for mat in self.mixing_mat_dict.values():
            condition_numbers.append(_compute_condition_number(mat, p=p))
        return condition_numbers


class RPS(_AbstractPS):
    """
    Class for Phase-based Separation method using real signals (and 2D-DFT).

    Parameters
    ----------
    N : int
        Number of nonlinear orders (truncation order of the Volterra series).
    nb_phase : int
        Number of phase factors used; should be greater than ``2*N+1``;
        choosing `nb_phase` large leads to a more robust method but also to
        more test signals.
    constant_term : boolean, optional (default=False)
        If True, constant term of the Volterra series (i.e. the order 0) is
        also separated; otherwise it is considered null.

    Attributes
    ----------
    N : int
        Truncation order.
    K : int
        Number of tests signals.
    nb_phase : int
        Number of phase factors.
    factors : array_like
        Vector of length `K` regrouping all factors.
    rho : float, class attribute
        Rejection factor; equal to None for RPS.
    w : float
        Initial phase factor.
    fft_axis : int or tuple(int), class attribute
        Axis along which inverse FFT is computed; equal to (0, 1) for RPS.
    mixing_mat_dict : dict(int: numpy.ndarray)
        Dictionnary of mixing matrix between orders and output for each phase.
    nq_dict : dict(int: list((int, int)))
        Dictionnary of list of tuples (n, q) for each phase.
    constant_term : False
        Whether constant term of the Volterra series is separated; have no
        effect for HPS.

    Methods
    -------
    gen_inputs(signal, return_cplx_sig=False)
        Returns the collection of input test signals.
    process_outputs(output_coll, raw_mode=False, N=None, constant_term=None)
        Process outputs and returns estimated orders or interconjugate terms.
    get_condition_numbers(p=2)
        Return the list of all condition numbers in the separation method.

    See also
    --------
    _AbstractPS: Parents class
    HPS, CPS, _SeparationMethod
    """

    fft_axis = (0, 1)
    args_diag = {'axis1': 0, 'axis2': 1}
    args_moveaxis = {'source': -1, 'destination': 0}

    def __init__(self, N, nb_phase=None, **kwargs):
        super().__init__(N, nb_phase=nb_phase, **kwargs)

        factors = []
        for w1, w2 in itr.combinations_with_replacement(self.factors, 2):
            # here we use a gain factor 1/2 instead of 2 in order to have
            # a maximum factor at 1 (and not 4)
            factors.append((1/2) * (w1 + w2))

        self.factors = factors
        self.K = len(factors)
        self._create_nq_dict()
        self._create_mixing_matrix_dict()

    @inherit_docstring
    def _create_tmp_mixing_matrix(self, phase):
        len_diag = self.N + int((phase % 2) == (self.N % 2))
        p1_start = ((phase + (self.N+1) % 2) // 2) - (self.N // 2)
        p1_end = p1_start + len_diag
        p1_vec = np.arange(p1_start, p1_end)
        tmp_mixing_mat = np.zeros((len_diag, len(self.nq_dict[phase])))

        # Computation of the interconjugate factor for each term
        for indp, (p1, p2) in enumerate(zip(p1_vec, p1_vec[::-1])):
            for indn, (n, q) in enumerate(self.nq_dict[phase]):
                if (abs(p1) + abs(p2)) <= n:
                    tmp_start = max(0, p1)
                    tmp_end = 1 + (n - abs(p2) + p1) // 2
                    k1_vec = np.arange(tmp_start, tmp_end)
                    k2_vec = k1_vec - p1
                    k3_vec = (n + p2 - (k1_vec + k2_vec)) // 2
                    k4_vec = k3_vec - p2

                    for k in zip(k1_vec, k2_vec, k3_vec, k4_vec):
                        tmp_mixing_mat[indp, indn] += multinomial(n, k) / 2**n

        return tmp_mixing_mat

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
        spectrum_2d = sc_fft.ifft2(output_coll_2d, axes=self.fft_axis)
        spectrum_2d = sc_fft.fftshift(spectrum_2d, axes=self.fft_axis)

        diff = self.nb_phase - self._compute_required_nb_phase()
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
            upper_diag = np.diagonal(out_per_phase, offset=phase,
                                     **self.args_diag)
            lower_diag = np.diagonal(out_per_phase, offset=-phase,
                                     **self.args_diag)
            upper_diag = np.moveaxis(upper_diag, **self.args_moveaxis)
            lower_diag = np.moveaxis(lower_diag, **self.args_moveaxis)
            return np.concatenate((np.real(upper_diag[slice_obj]),
                                   np.real(lower_diag[slice_obj]),
                                   np.imag(upper_diag[slice_obj]),
                                   np.imag(lower_diag[slice_obj])))
        else:
            temp = np.diagonal(out_per_phase, **self.args_diag)
            temp = np.moveaxis(temp, **self.args_moveaxis)
            return np.real(temp[slice_obj])

    def _fft_mat_condition_numbers(self, p):
        if p in [None, 'fro', 2, -2]:
            return 1
        if p in [np.inf, -np.inf, 1, -1]:
            return self.nb_phase**2


class PAS(_AbstractPS, AS):
    """
    Class for Phase-and-Amplitude-based Separation method using real signals.

    Parameters
    ----------
    N : int
        Number of nonlinear orders (truncation order of the Volterra series).
    nb_phase : int
        Number of phase factors used; should be greater than ``2*N+1``;
        choosing `nb_phase` large leads to a more robust method but also to
        more test signals.
    amplitudes : array_like, optional (default=None)
        Amplitude factors; if not specified, those values will be computed
        using parameters `gain` and `nb_amp` (if they are given), or chosen as
        the Chebyshev nodes.
    gain : float, optional (default=None)
        Gain factor in amplitude between the input test signals; if
        `amplitudes` is specified, this value is not used.
    nb_amp : int, optional (default=None)
        Number of different amplitudes; must be greater than or equal to
        ``(N + int(constant_term) + 1) // 2``; if None, will be set equal to
        ``(N + int(constant_term) + 1) // 2``; if `amplitudes` is specified,
        this value is not used.
    constant_term : boolean, optional (default=False)
        If True, constant term of the Volterra series (i.e. the order 0) is
        also separated; otherwise it is considered null.

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
        Vector of length `K` regrouping all factors.
    amplitudes : array_like
        Vector of length `nb_amp` regrouping all amplitudes.
    rho : float, class attribute
        Rejection factor; equal to None for PAS.
    w : float
        Initial phase factor.
    fft_axis : int or tuple(int), class attribute
        Axis along which inverse FFT is computed; equal to 0 for PAS.
    mixing_mat_dict : dict(int: numpy.ndarray)
        Dictionnary of mixing matrix between orders and output for each phase.
    nq_dict : dict(int: list((int, int)))
        Dictionnary of list of tuples (n, q) for each phase.
    constant_term : False
        Whether constant term of the Volterra series is separated; have no
        effect for HPS.

    Methods
    -------
    gen_inputs(signal, return_cplx_sig=False)
        Returns the collection of input test signals.
    process_outputs(output_coll, raw_mode=False)
        Process outputs and returns estimated orders or interconjugate terms.
    get_condition_numbers(p=2)
        Return the list of all condition numbers in the separation method.

    Class methods
    -------------
    optimum_gain(N, p=2, gain_min=.1, gain_max=.99, gain_init=None, tol=1e-6,
                 **kwargs)
        Search for the gain that minimizes the maximum condition number.

    See also
    --------
    _AbstractPS, AS: Parents class
    HPS, CPS, _SeparationMethod
    """

    def __init__(self, N, nb_phase=None, amplitudes=None, gain=None,
                 nb_amp=None, **kwargs):
        AS.__init__(self, N, amplitudes=amplitudes, gain=gain, nb_amp=nb_amp,
                    negative_amp=False, **kwargs)
        self.nb_phase = self._compute_required_nb_phase()
        self.HPS_obj = HPS(N, nb_phase=nb_phase)

        self._global_mix_mat = \
            _create_vandermonde_mixing_mat(self.factors, N,
                                           first_column=self.constant_term)
        self.nb_term = self.nb_amp * self.nb_phase

        factors = np.tensordot(self.HPS_obj.factors, self.factors, axes=0)
        self._update_factors(factors.flatten())

        self._create_nq_dict()
        self._create_mixing_matrix_dict()

    @inherit_docstring
    def _compute_required_nb_amp(self):
        return (self._N + 1) // 2

    @inherit_docstring
    def _create_tmp_mixing_matrix(self, phase):
        current_orders_index = [n-self._dec for (n, q) in self.nq_dict[phase]]
        tmp_mixing_mat = self._global_mix_mat[:, current_orders_index]
        for ind, (n, q) in enumerate(self.nq_dict[phase]):
            tmp_mixing_mat[:, ind] *= binomial(n, q)
        return tmp_mixing_mat

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

    def _fft_mat_condition_numbers(self, p):
        return self.HPS_obj._fft_mat_condition_numbers(p)
