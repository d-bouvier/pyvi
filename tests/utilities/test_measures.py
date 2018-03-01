# -*- coding: utf-8 -*-
"""
Test script for pyvi/utilities/measures.py

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import unittest
import numpy as np
from pyvi.utilities.measures import (separation_error, identification_error)


#==============================================================================
# Test Class
#==============================================================================

class SeparationErrorTestCase(unittest.TestCase):

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
                error = separation_error(self.sig, sig_est, db=False)
                self.assertEqual(len(error), self.N)

    def test_output_len_with_db_mode_on(self):
        for i, sigma in enumerate(self.sigma_values):
            with self.subTest(i=i):
                sig_est = self.sig + np.random.normal(scale=sigma,
                                                      size=self.size)
                error = separation_error(self.sig, sig_est, db=True)
                self.assertEqual(len(error), self.N)


class IdentificationErrorTest(unittest.TestCase):

    def setUp(self):
        self.N = 3
        self.M = 20
        self.kernels = dict()
        for n in range(1, self.N+1):
            self.kernels[n] = np.random.uniform(size=(self.M,)*n)
        self.sigma_values = [0, 0.001, 0.01, 0.1, 1]

    def test_output_len_with_db_mode_off(self):
        for i, sigma in enumerate(self.sigma_values):
            with self.subTest(i=i):
                kernels_est = dict()
                for n, h in self.kernels.items():
                    kernels_est[n] = h + np.random.normal(scale=sigma,
                                                          size=(self.M,)*n)
                error = identification_error(self.kernels, kernels_est,
                                             db=False)
                self.assertEqual(len(error), self.N)

    def test_output_len_with_db_mode_on(self):
        for i, sigma in enumerate(self.sigma_values):
            with self.subTest(i=i):
                kernels_est = dict()
                for n, h in self.kernels.items():
                    kernels_est[n] = h + np.random.normal(scale=sigma,
                                                          size=(self.M,)*n)
                error = identification_error(self.kernels, kernels_est,
                                             db=True)
                self.assertEqual(len(error), self.N)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
