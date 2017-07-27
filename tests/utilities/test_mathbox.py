# -*- coding: utf-8 -*-
"""
Test script for pyvi/utilities/mathbox.py

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 27 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import argparse
import numpy as np
from pyvi.utilities.mathbox import (rms, db, safe_db, binomial,
                                    array_symmetrization)
from mytoolbox.utilities.misc import my_parse_arg_for_tests


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    indent = my_parse_arg_for_tests()


    #########################
    ## Function binomial() ##
    #########################

    print(indent + 'Testing binomial()...', end=' ')
    for n in range(1, 10):
        assert binomial(n, 0) == 1, 'Wrong result for ({}, {}).'.format(n, 0)
        assert binomial(n, 1) == n, 'Wrong result for ({}, {}).'.format(n, 1)
        assert binomial(n, n) == 1, 'Wrong result for ({}, {}).'.format(n, n)
        for k in range(1, n):
            assert binomial(n, k) == binomial(n-1, k-1) + binomial(n-1, k), \
                'Wrong result for ({}, {})'.format(n, k)
    print('Done.')


    #####################################
    ## Function array_symmetrization() ##
    #####################################

    array = np.array([[1, 2, 4],
                      [0, 3, 6],
                      [0, 0, 8]])
    array_sym = np.array([[1, 1, 2],
                          [1, 3, 3],
                          [2, 3, 8]])
    array_sym_est = array_symmetrization(array)
    print(indent + 'Testing array_symmetrization()...', end=' ')
    assert np.all(array_sym == array_sym_est), 'Wrong result.'
    print('Done.')


    ####################
    ## Function rms() ##
    ####################

    array = np.arange(9).reshape(3, 3)
    rms_val = np.sqrt(np.mean(np.arange(9)**2))
    rms_val_axis0 = np.array([np.sqrt(np.mean(np.arange(0, 9, 3)**2)),
                              np.sqrt(np.mean(np.arange(1, 9, 3)**2)),
                              np.sqrt(np.mean(np.arange(2, 9, 3)**2))])
    rms_val_axis1 = np.array([np.sqrt(np.mean(np.arange(3)**2)),
                              np.sqrt(np.mean(np.arange(3, 6)**2)),
                              np.sqrt(np.mean(np.arange(6, 9)**2))])
    print(indent + 'Testing rms()...', end=' ')
    assert np.all(rms_val == rms(array)), 'Wrong result.'
    assert np.all(rms_val_axis0 == rms(array, axis=0)), 'Wrong result.'
    assert np.all(rms_val_axis1 == rms(array, axis=1)), 'Wrong result.'
    print('Done.')


    ##################################
    ## Functions db() and safe_db() ##
    ##################################

    vec = np.arange(-5, 6)
    sig = 10.**vec
    db_val_1 = 20*vec
    db_val_2 = db_val_1 - 20
    print(indent + 'Testing db()...', end=' ')
    assert np.all(db_val_1 == db(sig)), 'Wrong result.'
    assert np.all(db_val_1 == db(sig, ref=1.)), 'Wrong result.'
    assert np.all(db_val_2 == db(sig, ref=10.)), 'Wrong result.'
    print('Done.')
    print(indent + 'Testing safe_db()...', end=' ')
    assert np.all(db_val_1 == safe_db(sig, np.ones(sig.shape))), 'Wrong result.'
    assert np.all(db_val_2 == db(sig, 10*np.ones(sig.shape))), 'Wrong result.'
    print('Done.')