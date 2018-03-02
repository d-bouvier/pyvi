# -*- coding: utf-8 -*-
"""
Tools for kernel identification.

Functions
---------
assert_enough_data_samples :
    Assert that there is enough data samples for the identification.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""


#==============================================================================
# Functions
#==============================================================================

def assert_enough_data_samples(nb_data, max_nb_est, M, N, name):
    """
    Assert that there is enough data samples for the identification.

    Parameters
    ----------
    nb_data : int
        Number of data samples in the input signal used for identification.
    max_nb_est : int
        Maximum size of linear problem to solve.
    M : int or list(int)
        Memory length for each kernels (in samples).
    N : int
        Truncation order.
    name : str
        Name of the identification method.

    Raises
    ------
    ValueError
        If L is inferior to the number of Volterra coefficients.
    """

    if nb_data < max_nb_est:
        raise ValueError('Input signal has {} data samples'.format(nb_data) +
                         ', it should have at least {} '.format(max_nb_est) +
                         'for a truncation to order {} '.format(N) +
                         'and a {}-samples memory length'.format(M) +
                         'using {} method.'.format(name))
