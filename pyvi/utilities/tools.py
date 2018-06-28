# -*- coding: utf-8 -*-
"""
Miscellaneous tools.

Class
-----
DocInherit :
    Docstring inheriting method descriptor.

Functions
---------
_as_list :
    Check that given variable is a list or a tuple.
_is_sorted :
    Check that given numeric array is sorted in ascending order.

Decorator
---------
inherit_docstring :
    Decorator for making methods inherit docstrings.

Notes
-----
Developed for Python 3.6
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
`DocInherit` found on http://code.activestate.com/recipes/576862/
"""

#==============================================================================
# Importations
#==============================================================================

from functools import wraps
from collections.abc import Sequence
import numpy as np


#==============================================================================
# Class
#==============================================================================

class DocInherit(object):
    """
    Docstring inheriting method descriptor.

    The class itself is also used as a decorator.
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    #     return self.get(cls, obj)

    # def get(self, cls, instance):
    def __get__(self, obj, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        if obj:
            @wraps(self.mthd, assigned=('__name__', '__module__'))
            def f(*args, **kwargs):
                return self.mthd(obj, *args, **kwargs)
        else:
            @wraps(self.mthd, assigned=('__name__', ' __module__'))
            def f(*args, **kwargs):
                return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find {} in parents".format(self.name))
        func.__doc__ = source.__doc__
        return func


inherit_docstring = DocInherit


#==============================================================================
# Functions
#==============================================================================

def _as_list(val, N):
    """
    Check that given variable is a list or a tuple.

    Check that `val` is an instance of list, tuple or numpy.ndarray with
    length `N`. If not, this functions returns a list of length `N`, with
    every value being `val`.

    Parameters
    ----------
    M : type, list(type) or tuple(type)
        Variable to output as a list.
    N : int
        Truncation order.

    Returns
    -------
    list(type)
        List of length `N`.
    """

    if isinstance(val, (Sequence, np.ndarray)) and not isinstance(val, str):
        if len(val) != N:
            raise ValueError('`val` has length {}, but '.format(len(val)) +
                             'truncation order N is {}'.format(N))
        else:
            return list(val)
    else:
        return [val, ]*N


def _is_sorted(array_1d):
    """
    Check that given numeric array is sorted in ascending order.

    Parameters
    ----------
    array_1d : array_like
        Numeric array to check if it is sorted.

    Returns
    -------
    is_sorted : boolean
        True if `array_1d` is sorted, False otherwise.
    """

    for ind in range(len(array_1d)-1):
        if array_1d[ind+1] < array_1d[ind]:
            return False
    return True
