# -*- coding: utf-8 -*-
"""
Test script for pyvi/simulation/combinatorics.py

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 24 Nov. 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import unittest
import numpy as np
import pyvi.simulation.combinatorics as comb


#==============================================================================
# Test Class
#==============================================================================

class MakeListPqTestCase(unittest.TestCase):

    def test_output(self):
        N = 4
        computed_list = comb.make_list_pq(N)
        true_list = np.array([[2, 2, 0], [2, 1, 1], [2, 0, 2], [3, 2, 0],
                              [3, 1, 1], [3, 3, 0], [3, 2, 1], [3, 1, 2],
                              [3, 0, 3], [4, 2, 0], [4, 1, 1], [4, 3, 0],
                              [4, 2, 1], [4, 1, 2], [4, 4, 0], [4, 3, 1],
                              [4, 2, 2], [4, 1, 3], [4, 0, 4]])
        self.assertTrue(np.all(computed_list == true_list))


class EliminationCase(unittest.TestCase):

    def test_output(self):
        N = 4
        pq_dict = {(2, 0): 1, (1, 1): 1}
        computed_list = comb.elimination(pq_dict, comb.make_list_pq(N))
        true_list = np.array([[2, 2, 0], [2, 1, 1], [3, 2, 0], [3, 1, 1],
                              [4, 2, 0], [4, 1, 1]])
        self.assertTrue(np.all(computed_list == true_list))


class StateCombinatoricsCase(unittest.TestCase):

    def test_output_without_elimination(self):
        N = 4
        computed_list = comb.make_list_pq(N)
        true_pq_sets = {2: [(2, 0, (1, 1), 1), (1, 1, (1,), 1), (0, 2, (), 1)],
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
        computed_pq_sets = comb.state_combinatorics(computed_list, N, True)
        self.assertDictEqual(computed_pq_sets, true_pq_sets)

    def test_output_with_elimination(self):
        N = 4
        pq_dict = {(2, 0): 1, (1, 1): 1}
        computed_list = comb.elimination(pq_dict, comb.make_list_pq(N))
        true_pq_sets = {2: [(2, 0, (1, 1), 1), (1, 1, (1,), 1)],
                        3: [(2, 0, (1, 2), 2), (1, 1, (2,), 1)],
                        4: [(2, 0, (1, 3), 2), (2, 0, (2, 2), 1),
                            (1, 1, (3,), 1)]}
        computed_pq_sets = comb.state_combinatorics(computed_list, N, True)
        self.assertDictEqual(computed_pq_sets, true_pq_sets)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
