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

Decorator
---------
inherit_docstring :
    Decorator for making methods inherit docstrings.

Notes
-----
Developed for Python 3.6.1
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

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

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

    Check that `val` is an instance of list or tuple class with length `N`.
    If not, this functions returns a list of length `N`, with every value
    being `val`.

    Parameters
    ----------
    M : type, list(type) or tuple(type)
        Variable to output as a list.
    N : int
        Truncation order.

    Returns
    -------
    list : list(type)
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
