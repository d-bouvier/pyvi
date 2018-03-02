# -*- coding: utf-8 -*-
"""
Tools for handling Volterra series and kernels.

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
    Check that ``val`` is a list or a tuple of length ``N``.

    If ``val`` is not an instance of a list or a tuple, this functions returns
    a list of length ``N``, with every value being ``val`.

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
                             'truncation order N is {}'.format(val))
        else:
            return list(val)
    else:
        return [val, ]*N
