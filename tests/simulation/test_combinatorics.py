# -*- coding: utf-8 -*-
"""
Test script for pyvi.simulation.combinatorics

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 27 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import argparse
import numpy as np
import pyvi.simulation.combinatorics as comb


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    #####################
    ## Parsing options ##
    #####################

    parser = argparse.ArgumentParser()
    parser.add_argument('-ind', '--indentation', type=int, default=0)
    args = parser.parse_args()
    indent = args.indentation
    ss = ' ' * indent


    ##########
    ## Data ##
    ##########

    N = 4
    pq_dict = {(2, 0): 1,
               (1, 1): 1}

    list_pq = np.array([[2, 2, 0], [2, 1, 1], [2, 0, 2],
                        [3, 2, 0], [3, 1, 1],
                        [3, 3, 0], [3, 2, 1], [3, 1, 2], [3, 0, 3],
                        [4, 2, 0], [4, 1, 1],
                        [4, 3, 0], [4, 2, 1], [4, 1, 2],
                        [4, 4, 0], [4, 3, 1], [4, 2, 2], [4, 1, 3], [4, 0, 4]])
    list_pq_2 = np.array([[2, 2, 0], [2, 1, 1],
                          [3, 2, 0], [3, 1, 1],
                          [4, 2, 0], [4, 1, 1]])
    pq_sets = {2: [(2, 0, (1, 1), 1), (1, 1, (1,), 1), (0, 2, (), 1)],
               3: [(2, 0, (1, 2), 2),
                   (1, 1, (2,), 1),
                   (3, 0, (1, 1, 1), 1),
                   (2, 1, (1, 1), 1),
                   (1, 2, (1,), 1),
                   (0, 3, (), 1)],
               4: [(2, 0, (1, 3), 2),
                   (2, 0, (2, 2), 1),
                   (1, 1, (3,), 1),
                   (3, 0, (1, 1, 2), 3),
                   (2, 1, (1, 2), 2),
                   (1, 2, (2,), 1),
                   (4, 0, (1, 1, 1, 1), 1),
                   (3, 1, (1, 1, 1), 1),
                   (2, 2, (1, 1), 1),
                   (1, 3, (1,), 1),
                   (0, 4, (), 1)]}
    pq_sets_2 = {2: [(2, 0, (1, 1), 1), (1, 1, (1,), 1)],
                 3: [(2, 0, (1, 2), 2), (1, 1, (2,), 1)],
                 4: [(2, 0, (1, 3), 2), (2, 0, (2, 2), 1), (1, 1, (3,), 1)]}


    ################################################
    ## Functions of pyvi.simulation.combinatorics ##
    ################################################

    print(ss + 'Testing make_list_pq()...', end=' ')
    list_pq_computed = comb.make_list_pq(N)
    list_pq_2_computed = comb.elimination(pq_dict, list_pq_computed)
    assert np.all(list_pq == list_pq_computed), 'Error in make_list_pq().'
    print('Done.')

    print(ss + 'Testing elimination()...', end=' ')
    assert np.all(list_pq_2 == list_pq_2_computed), 'Error in elimination().'
    print('Done.')

    print(ss + 'Testing state_combinatorics()...', end=' ')
    pq_sets_computed = comb.state_combinatorics(list_pq_computed, N, True)
    pq_sets_2_computed = comb.state_combinatorics(list_pq_2_computed, N, True)
    assert np.all(pq_sets == pq_sets_computed), \
        'Error in state_combinatorics().'
    assert np.all(pq_sets_2 == pq_sets_2_computed), \
        'Error in state_combinatorics().'
    print('Done.')