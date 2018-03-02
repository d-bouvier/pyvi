# -*- coding: utf-8 -*-
"""
Miscellaneous tools.

Functions
---------
_as_list :
    Check that given variable is a list or a tuple.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""


#==============================================================================
# Functions
#==============================================================================

def _as_list(val, N):
    """
    Check that given variable is a list or a tuple.

    Check that ``val`` is an instance of list or tuple class with length ``N``.
    If not, this functions returns a list of length ``N``, with every value
    being ``val`.

    Parameters
    ----------
    M : type, list(type) or tuple(type)
        Variable to output as a list.
    N : int
        Truncation order.

    Returns
    -------
    list : list(type)
        List of length ``N``.
    """

    if isinstance(val, list) or isinstance(val, tuple):
        if len(val) != N:
            raise ValueError('``val` has length {}, but '.format(len(val)) +
                             'truncation order N is {}'.format(N))
        else:
            return list(val)
    else:
        return [val, ]*N
