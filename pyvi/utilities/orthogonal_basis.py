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

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

from numbers import Number
from collections.abc import Sequence
import numpy as np
import scipy.signal as sc_sig
from .tools import inherit_docstring


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
        projection : array_like
            Signal projection; projected elements are along the first axis.
        """

        return np.zeros((self.K,) + signal.shape, signal.dtype)


    @classmethod
    def _filtering(cls, signal, system):
        """Filter `signal` by `system`."""

        if any(np.iscomplex(signal)):
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

    Attributes
    ----------
    pole : float
    K : int

    Methods
    -------
    projection(signal)
        Project a signal unto the basis.
    """

    def __init__(self, pole, K):
        if np.iscomplex(pole):
            raise ValueError('Given parameter `pole` is complex-valued, ' +
                             'should be real-valued for Laguerre basis.')
        self.pole = pole
        self.K = K
        self._init_filter, self._post_filter = self._compute_filters(pole)

    @classmethod
    def _compute_filters(cls, pole):
        init_filt = sc_sig.dlti([], [pole], np.sqrt(1 - pole**2))
        post_filt = sc_sig.dlti([1/pole], [pole], pole)
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

    Attributes
    ----------
    pole : float
    K : int

    Methods
    -------
    projection(signal)
        Project a signal unto the basis.
    """

    def __init__(self, pole, K):
        if K % 2:
            raise ValueError('Given parameter `K` is odd, should be even ' +
                             'for Kautz basis to ensure realness.')
        self.pole = pole
        self.K = K
        filters = self._compute_filters(pole)
        self._init_filter, self._even_filter, self._post_filter = filters

    @classmethod
    def _compute_filters(cls, pole):
        c = - np.abs(pole)**2
        b = 2 * np.real(pole) / (1 - c)
        gain_odd = np.sqrt(1 - c**2)
        gain_even = np.sqrt(1 - b**2)
        num = [-c, b*(c-1), 1]
        den = [1, b*(c-1), -c]

        init_filt = sc_sig.dlti([gain_odd, -gain_odd*b], den)
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

    Attributes
    ----------
    poles : list(float or complex)
        List of all the poles, including complex conjuguated ones.
    K : int
        Number of elements of the basis.

    Methods
    -------
    projection(signal)
        Project a signal unto the basis.
    """

    def __init__(self, poles):
        self.poles = []
        self._filters = []
        self._type_list = []
        for pole in poles:
            if np.iscomplex(pole):
                self.poles += [pole, np.conj(pole)]
                self._filters.append(KautzBasis._compute_filters(pole))
                self._type_list.append('Kautz')
            else:
                self.poles.append(pole)
                self._filters.append(LaguerreBasis._compute_filters(pole))
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

def create_orthogonal_basis(poles, K=None):
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

    Returns
    -------
    basis: LaguerreBasis, KautzBasis, GeneralizedBasis
        Returned orthogonal basis; its type depends on the given parameters.
    """

    if isinstance(poles, (Sequence, np.ndarray)):
        if len(poles) == 0:
            raise ValueError('Parameter `poles` has zero-length, should ' +
                             'be at least 1.')
        elif len(poles) == 1:
            return create_orthogonal_basis(poles[0], K=K)
        else:
            return GeneralizedBasis(poles)
    elif isinstance(poles, Number):
        if K is None:
            raise ValueError('Unspecified parameter `K` for basis of ' +
                             'type Laguerre or Kautz.')
        pole = poles
        if np.iscomplex(pole):
            return KautzBasis(pole, K)
        else:
            return LaguerreBasis(np.real(pole), K)
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
