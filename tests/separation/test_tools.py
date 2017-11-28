# -*- coding: utf-8 -*-
"""
Test script for pyvi/separation/tools.py

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 27 Nov. 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import unittest
import numpy as np
from pyvi.separation.tools import error_measure


#==============================================================================
# Test Class
#==============================================================================

class ErrorMeasureTestCase(unittest.TestCase):

    def setUp(self):
        self.N = 3
        self.L = 1000
        self.size = (self.N, self.L)
        self.sig = np.random.uniform(low=-1.0, high=1.0, size=self.size)
        self.sigma_values = [0, 0.001, 0.01, 0.1, 1]

    def test_output_len_with_db_mode_off(self):
        for i, sigma in enumerate(self.sigma_values):
            with self.subTest(i=i):
                sig_est = self.sig + np.random.normal(scale=sigma,
                                                      size=self.size)
                error = error_measure(self.sig, sig_est, db=False)
                self.assertEqual(len(error), self.N)

    def test_output_len_with_db_mode_on(self):
        for i, sigma in enumerate(self.sigma_values):
            with self.subTest(i=i):
                sig_est = self.sig + np.random.normal(scale=sigma,
                                                      size=self.size)
                error = error_measure(self.sig, sig_est, db=True)
                self.assertEqual(len(error), self.N)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
