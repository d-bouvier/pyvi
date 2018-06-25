# -*- coding: utf-8 -*-
"""
Module for orthogonal basis creation and projection.

This modules creates class to handle orthogonal basis for signal projection.
A valid basis object is:

    - an instance of a subclass of :class:`_OrthogonalBasis`, such as
    :class:`LaguerreBasis`, :class:`KautzBasis` or :class:`GeneralizedBasis`;
    - an instance of a custom object such that the following conditions are
    met:

        - ``hasattr(basis, 'K') == True``;
        - ``hasattr(basis, 'projection') == True``;
        - ``callable(getattr(basis, 'projection', None)) == True``;
        - ``isinstance(basis.projection(signal), numpy.ndarray) == True``
        with ``isinstance(signal, numpy.ndarray)``;
        - ``basis.projection(signal).shape == (basis.K,) + shape``
        with ``shape = signal.shape``.

    Those conditions can be checked using :func:`is_valid_basis_instance()`.

Class
-----
LaguerreBasis :
    Class for Orthogonal Laguerre Basis.
KautzBasis :
    Class for Orthogonal Kautz Basis.
GeneralizedBasis :
    Class for Generalized Orthogonal Basis.

Functions
---------
create_orthogonal_basis :
    Returns an orthogonal basis given its poles and its number of elements.
is_valid_basis_instance :
    Checks whether `basis` is a usable instance of a basis.
laguerre_pole_optimization :
    Compute the optimized Laguerre pole from the Laguerre spectra of a kernel.

Notes
-----
Developed for Python 3.6
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

__all__ = ['LaguerreBasis', 'KautzBasis', 'GeneralizedBasis',
           'create_orthogonal_basis', 'is_valid_basis_instance']


#==============================================================================
# Importations
#==============================================================================

from numbers import Number
from collections.abc import Sequence
import itertools as itr
import numpy as np
import scipy.signal as sc_sig
from .tools import inherit_docstring, _is_sorted


#==============================================================================
# Class
#==============================================================================

class _OrthogonalBasis():
    """
    Abstract class for orthogonal basis.
    """

    def projection(self, signal):
        """
        Project a signal unto the basis.

        Parameters
        ----------
        signal : array_like
            Signal to project unto the orthogonal basis.

        Returns
        -------
        array_like
            Signal projection; projected elements are along the first axis.
        """

        return np.zeros((self.K,) + signal.shape, signal.dtype)


    @classmethod
    def _filtering(cls, signal, system):
        """Filter `signal` by `system`."""

        if np.iscomplexobj(signal):
            _, filtered_signal_r, _ = sc_sig.dlsim(system, np.real(signal))
            _, filtered_signal_i, _ = sc_sig.dlsim(system, np.imag(signal))
            filtered_signal = filtered_signal_r + 1j * filtered_signal_i
        else:
            _, filtered_signal, _ = sc_sig.dlsim(system, signal)
        filtered_signal.shape = signal.shape
        return filtered_signal


class LaguerreBasis(_OrthogonalBasis):
    """
    Class for Orthogonal Laguerre Basis.

    Parameters
    ----------
    pole : float
        Real-valued Laguerre pole.
    K : int
        Number of elements of the basis.
    unit_delay : boolean, optional (default=False)
        If True, the filters are all strictly causal (they verify the unit
        delay condition).

    Attributes
    ----------
    pole : float
    K : int
    _unit_delay : boolean

    Methods
    -------
    projection(signal)
        Project a signal unto the basis.
    """

    def __init__(self, pole, K, unit_delay=False):
        if np.iscomplex(pole):
            raise ValueError('Given parameter `pole` is complex-valued, ' +
                             'should be real-valued for Laguerre basis.')
        self.pole = pole
        self.K = K
        self._unit_delay = unit_delay
        self._init_filter, self._post_filter = \
            self._compute_filters(pole, unit_delay)

    @classmethod
    def _compute_filters(cls, pole, unit_delay):
        if pole == 0:
            poles_init = [0] if unit_delay else []
            init_filt = sc_sig.dlti([], poles_init, 1)
            post_filt = sc_sig.dlti([], [pole], 1)
        else:
            zeros_init = [] if unit_delay else [0]
            init_filt = sc_sig.dlti(zeros_init, [pole], np.sqrt(1 - pole**2))
            post_filt = sc_sig.dlti([1/pole], [pole], -pole)
        return init_filt._as_ss(), post_filt._as_ss()

    @inherit_docstring
    def projection(self, signal):
        projection = super().projection(signal)
        current_sig = self._filtering(signal, self._init_filter)
        for k in range(self.K):
            projection[k] = current_sig.copy()
            current_sig = self._filtering(current_sig, self._post_filter)
        return projection


class KautzBasis(_OrthogonalBasis):
    """
    Class for Orthogonal Kautz Basis.

    Parameters
    ----------
    pole : complex
        Complex-valued Kautz pole.
    K : int
        Number of elements of the basis.
    unit_delay : boolean, optional (default=False)
        If True, the filters are all strictly causal (they verify the unit
        delay condition).

    Attributes
    ----------
    pole : float
    K : int
    _unit_delay : boolean

    Methods
    -------
    projection(signal)
        Project a signal unto the basis.
    """

    def __init__(self, pole, K, unit_delay=False):
        if K % 2:
            raise ValueError('Given parameter `K` is odd, should be even ' +
                             'for Kautz basis to ensure realness.')
        self.pole = pole
        self.K = K
        self._unit_delay = unit_delay
        filters = self._compute_filters(pole, unit_delay)
        self._init_filter, self._even_filter, self._post_filter = filters

    @classmethod
    def _compute_filters(cls, pole, unit_delay):
        if pole == 0:
            den_init = [1]
            if unit_delay:
                den_init.append(0)
            init_filt = sc_sig.dlti([1], den_init)
            even_filt = sc_sig.dlti([1], [1, 0])
            post_filt = sc_sig.dlti([1], [1, 0, 0])
        else:
            c = - np.abs(pole)**2
            b = 2 * np.real(pole) / (1 - c)
            gain_odd = np.sqrt(1 - c**2)
            gain_even = np.sqrt(1 - b**2)
            num = [-c, b*(c-1), 1]
            den = [1, b*(c-1), -c]

            num_init = [gain_odd, -gain_odd*b]
            if not unit_delay:
                num_init.append(0)

            init_filt = sc_sig.dlti(num_init, den)
            even_filt = sc_sig.dlti([gain_even], [1, -b])
            post_filt = sc_sig.dlti(num, den)
        return init_filt._as_ss(), even_filt._as_ss(), post_filt._as_ss()

    @inherit_docstring
    def projection(self, signal):
        projection = super().projection(signal)
        current_sig = self._filtering(signal, self._init_filter)
        for k in range(self.K//2):
            projection[2*k] = current_sig.copy()
            projection[2*k+1] = self._filtering(current_sig, self._even_filter)
            current_sig = self._filtering(current_sig, self._post_filter)
        return projection


class GeneralizedBasis(_OrthogonalBasis):
    """
    Class for Generalized Orthogonal Basis.

    Parameters
    ----------
    poles : list(float or complex)
        List of the wanted poles; for complex poles, conjuguated poles are
        automatically added.
    unit_delay : boolean, optional (default=False)
        If True, the filters are all strictly causal (they verify the unit
        delay condition).

    Attributes
    ----------
    poles : list(float or complex)
        List of all the poles, including complex conjuguated ones.
    K : int
        Number of elements of the basis.
    _unit_delay : boolean

    Methods
    -------
    projection(signal)
        Project a signal unto the basis.
    """

    def __init__(self, poles, unit_delay=False):
        self._unit_delay = unit_delay
        self.poles = []
        self._filters = []
        self._type_list = []
        for pole in poles:
            if np.iscomplex(pole):
                self.poles += [pole, np.conj(pole)]
                self._filters.append(
                    KautzBasis._compute_filters(pole, unit_delay))
                self._type_list.append('Kautz')
            else:
                self.poles.append(pole)
                self._filters.append(
                    LaguerreBasis._compute_filters(pole, unit_delay))
                self._type_list.append('Laguerre')
        self.K = len(self.poles)

    @inherit_docstring
    def projection(self, signal):
        projection = super().projection(signal)
        current_sig = signal.copy()
        k = 0
        for step_type, step_filters in zip(self._type_list, self._filters):
            if step_type == 'Kautz':
                tmp_sig = self._filtering(current_sig, step_filters[0])
                projection[k] = tmp_sig
                projection[k+1] = self._filtering(tmp_sig, step_filters[1])
                current_sig = self._filtering(current_sig, step_filters[2])
                k += 2
            else:
                projection[k] = self._filtering(current_sig, step_filters[0])
                current_sig = self._filtering(current_sig, step_filters[1])
                k += 1
        return projection


#==============================================================================
# Functions
#==============================================================================

def create_orthogonal_basis(poles, K=None, unit_delay=False):
    """
    Returns an orthogonal basis given its poles and its number of elements.

    Parameters
    ----------
    poles : number or list(number)
        Poles defining the orthogonal basis; if only one real (respectively
        complex) pole is given, a Laguerre (resp. Kautz) basis is returned; if
        several poles are given, a Generalized orthogonal basis is returned.
    K : int, optional (default=None)
        Number of elements of the basis; only mandatory if `poles` is a
        number or of lenght 1; else, the number of elements will depend of the
        number of poles.
    unit_delay : boolean, optional (default=False)
        If True, the filters are all strictly causal (they verify the unit
        delay condition).

    Returns
    -------
    LaguerreBasis, KautzBasis, GeneralizedBasis
        Returned orthogonal basis; its type depends on the given parameters.
    """

    if isinstance(poles, (Sequence, np.ndarray)):
        if len(poles) == 0:
            raise ValueError('Parameter `poles` has zero-length, should ' +
                             'be at least 1.')
        elif len(poles) == 1:
            return create_orthogonal_basis(poles[0], K=K,
                                           unit_delay=unit_delay)
        else:
            return GeneralizedBasis(poles, unit_delay=unit_delay)
    elif isinstance(poles, Number):
        if K is None:
            raise ValueError('Unspecified parameter `K` for basis of ' +
                             'type Laguerre or Kautz.')
        pole = poles
        if np.iscomplex(pole):
            return KautzBasis(pole, K, unit_delay=unit_delay)
        else:
            return LaguerreBasis(np.real(pole), K, unit_delay=unit_delay)
    else:
        raise TypeError('Parameter `poles` is neither a numeric value ' +
                        'nor a list of numeric values.')


def is_valid_basis_instance(basis):
    """Checks whether `basis` is a usable instance of a basis."""

    sig = np.sin(2 * np.pi * np.arange(10)/10)
    shape = sig.shape

    try:
        conditions = [hasattr(basis, 'K'), hasattr(basis, 'projection'),
                      callable(getattr(basis, 'projection', None)),
                      isinstance(basis.projection(sig), np.ndarray),
                      basis.projection(sig).shape == (basis.K,)+shape]
    except:
        return False

    return all(conditions)


def laguerre_pole_optimization(pole, projection, n, nb_base, form=None,
                               return_cost=False):
    """
    Compute the optimized Laguerre pole from the Laguerre spectra of a kernel.

    Use the method described in [1] to find an optimal value of the Laguerre
    pole for the projection of a given Volterra kernel (known under its
    Laguerre spectra form); due to the truncation of the Laguerre basis, the
    method will not find the optimal value, and thus should be applied
    iteratively.

    Parameters
    ----------
    pole : float
        Current value of the Laguerre pole; should be between -1 and 1.
    projection : np.ndarray
        Laguerre spectra of the kernel estimated using value `pole` as
        Laguerre pole.
    n : int
        Order of the kernel.
    nb_base : int
        Number of element in the Laguerre basis used for the expansion.
    form : {'vector', 'kernel', None}, optional (default=None)
        Form under which the projection is given; if None, the form is found
        from `projection`.
    return_cost : boolean, (optional=False)
        If True, this function also returns the cost after optimization.

    Returns
    -------
    new_pole : float
        New Laguerre pole value; is between -1 and 1.
    cost : float
        Value of the cost after optimization (see [1]); only returned if
        `cost` is True.

    References
    ----------
    .. [1] A. Kibangou, G. Favier, M. Hassani "Laguerre-Volterra Filters
       Optimization Based on Laguerre Spectra", EURASIP Journal on Applied
       Signal Processing, Computers & Geosciences, vol. 17, pp. 2874-2887,
       2005.
    """

    if form is None:
        value = np.squeeze(projection).ndim
        if not value:
            raise ValueError
        if value == 1:
            form = 'vector'
        else:
            form = 'kernel'

    if form == 'vector':
        ind_mat = _compute_ind_mat(n, nb_base, len(projection))
        R1 = _compute_R1_from_vector(n, ind_mat, projection)
        R2 = _compute_R2_from_vector(n, ind_mat, projection)
    else:
        R1 = _compute_R1_from_kernel(n, nb_base, projection)
        R2 = _compute_R2_from_kernel(n, nb_base, projection)

    rho = ((1+pole**2) * R1 + 2*pole*R2) / (2*pole*R1 + (1+pole**2)*R2)
    if rho > 1:
        new_pole = rho - np.sqrt(rho**2 - 1)
    elif rho <= -1:
        new_pole = rho + np.sqrt(rho**2 - 1)

    if return_cost:
        norm_l2 = _compute_norm_l2(projection)

        den = 2 * (1-pole**2) * n * norm_l2
        a = 1 + pole**2
        b = 2 * pole
        Q1 = (a*R1 + b*R2)/den - (1/2)
        Q2 = (b*R1 + a*R2)/den

        num = (1+Q1) * pole**2 - 2 * Q2 * pole + Q1
        cost = num / (1 - pole**2)
        return new_pole, cost
    else:
        return new_pole


def _compute_ind_mat(n, m, nb_coeff):
    """Compute matrix of indexes for each coefficient."""

    ind_mat = np.zeros((nb_coeff, n))
    curr_idx = 0
    for indexes in itr.combinations_with_replacement(range(m), n):
        ind_mat[curr_idx] = np.array(indexes)
        curr_idx += 1

    return ind_mat


def _compute_R1_from_vector(n, ind_mat, vec):
    """Compute term R1 from projection under vector form."""

    R1 = 0
    for l in range(n):
        R1 += np.sum((2*ind_mat[:, l]+1) * vec**2)

    return R1


def _compute_R1_from_kernel(n, m, kernel):
    """Compute term R1 from projection under kernel form."""

    R1 = 0
    ind_vec = np.arange(m)
    for l in range(n):
        ind_vec.shape = (1,)*l + (m,) + (1,)*(n-l-1)
        R1 += np.sum((2*ind_vec+1) * kernel**2)

    return R1


def _compute_R2_from_vector(n, ind_mat, vec):
    """Compute term R2 from projection under vector form."""

    R2 = 0
    for l in range(n):
        _idx2keep = np.where(ind_mat[:, l] > 0)[0]
        idx2keep_1 = []
        idx2keep_2 = []
        for idx in _idx2keep:
            temp = ind_mat[idx, :].copy()
            temp[l] -= 1
            if _is_sorted(temp):
                res_temp = np.where((ind_mat == temp).all(axis=1))
                idx2keep_1.append(idx)
                idx2keep_2.append(res_temp[0][0])
        temp_vec = vec[idx2keep_1] * vec[idx2keep_2]
        R2 = 2 * np.sum(ind_mat[idx2keep_1, l] * temp_vec)

    return R2


def _compute_R2_from_kernel(n, m, kernel):
    """Compute term R2 from projection under kernel form."""

    R2 = 0
    ind_vec = np.arange(m)
    for l in range(n):
        ind_vec.shape = (1,)*l + (m,) + (1,)*(n-l-1)
        _idx1 = (slice(None),)*l + (slice(1, None),) + (slice(None),)*(n-l-1)
        _idx2 = (slice(None),)*l + (slice(m-1),) + (slice(None),)*(n-l-1)
        R2 += 2 * np.sum(ind_vec[_idx1] * kernel[_idx1] * kernel[_idx2])

    return R2


def _compute_norm_l2(kernel):
    return np.sum(kernel**2)
