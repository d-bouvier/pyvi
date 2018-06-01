# -*- coding: utf-8 -*-
"""
Test script for pyvi/separation/tools.py

Notes
-----
Developed for Python 3.6
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import unittest
import numpy as np
from pyvi.separation.tools import _create_vandermonde_mixing_mat


#==============================================================================
# Test Class
#==============================================================================

class CreateVandermondeMixingMatTest(unittest.TestCase):

    factors = [1, 2, 3]
    N = 3
    results = {True: np.array([[1, 1, 1, 1],
                               [1, 2, 4, 8],
                               [1, 3, 9, 27]]),
               False: np.array([[1, 1, 1],
                                [2, 4, 8],
                                [3, 9, 27]])}

    def test_correct(self):
        for test in [True, False]:
            with self.subTest(i=test):
                result = _create_vandermonde_mixing_mat(self.factors, self.N,
                                                        first_column=test)
                self.assertTrue(np.all(result == self.results[test]))


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
