# -*- coding: utf-8 -*-
"""
Test script for pyvi/identification/tools.py

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
from pyvi.identification.tools import (_assert_enough_data_samples,
                                       _complex2real)


#==============================================================================
# Test Class
#==============================================================================

class AssertEnoughDataSamplesTest(unittest.TestCase):

    def test_error_raised(self):
        self.assertRaises(ValueError, _assert_enough_data_samples, 8, 9,
                          3, 2, 'KLS')


class Complex2RealTest(unittest.TestCase):

    def setUp(self):
        self.val = np.array([1 + 2j, 3 + 4j])
        self.real = np.array([1, 3])
        self.imag = np.array([2, 4])
        self.real_imag = np.array([1, 3, 2, 4])

    def test_default_mode(self):
        result = _complex2real(self.val)
        self.assertTrue(np.all(result == self.real_imag))

    def test_real_mode(self):
        result = _complex2real(self.val, cast_mode='real')
        self.assertTrue(np.all(result == self.real))

    def test_imag_mode(self):
        result = _complex2real(self.val, cast_mode='imag')
        self.assertTrue(np.all(result == self.imag))

    def test_real_imag_mode(self):
        result = _complex2real(self.val, cast_mode='real-imag')
        self.assertTrue(np.all(result == self.real_imag))

    def test_cplx_mode(self):
        result = _complex2real(self.val, cast_mode='cplx')
        self.assertTrue(np.all(result == self.val))

    def test_warns(self):
        self.assertWarns(UserWarning, _complex2real, self.val, cast_mode='')


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
