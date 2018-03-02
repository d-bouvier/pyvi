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
from pyvi.identification.tools import (assert_enough_data_samples,
                                       volterra_basis)
from pyvi.volterra.tools import kernel_nb_coeff


#==============================================================================
# Test Class
#==============================================================================

class AssertEnoughDataSamplesTest(unittest.TestCase):

    def test_error_raised(self):
        self.assertRaises(ValueError, assert_enough_data_samples, 8, 9,
                          3, 2, 'KLS')


class VolterraBasisTest(unittest.TestCase):

    def setUp(self):
        self.L = 100
        self.N = 4
        self.M = 15
        self.M_2 = [10, 15, 0, 5]
        sig_r = np.arange(1, self.L+1)
        sig_c = np.arange(self.L) + 2j * np.arange(self.L)
        self.order_keys = {1: 0, 2: 0, 3: 0, 4: 0}
        self.order_r = volterra_basis(sig_r, self.M, self.N, mode='order')
        self.order_c = volterra_basis(sig_c, self.M, self.N, mode='order')
        self.order_r_2 = volterra_basis(sig_r, self.M_2, self.N, mode='order')
        self.order_c_2 = volterra_basis(sig_c, self.M_2, self.N, mode='order')
        self.term_keys = {(1, 0): 0, (2, 0): 0, (2, 1): 0, (3, 0): 0,
                          (3, 1): 0, (4, 0): 0, (4, 1): 0, (4, 2): 0}
        self.term_r = volterra_basis(sig_r, self.M, self.N, mode='term')
        self.term_c = volterra_basis(sig_c, self.M, self.N, mode='term')
        self.term_r_2 = volterra_basis(sig_r, self.M_2, self.N, mode='term')
        self.term_c_2 = volterra_basis(sig_c, self.M_2, self.N, mode='term')

    def test_output_type_for_orders(self):
        for i, value in enumerate([self.order_r, self.order_c, self.order_r_2,
                                   self.order_c_2]):
            with self.subTest(i=i):
                self.assertIsInstance(value, dict)

    def test_output_type_for_terms(self):
        for i, value in enumerate([self.term_r, self.term_c, self.term_r_2,
                                   self.term_c_2]):
            with self.subTest(i=i):
                self.assertIsInstance(value, dict)

    def test_output_shape_for_orders(self):
        for i, value in enumerate([self.order_r, self.order_c, self.order_r_2,
                                   self.order_c_2]):
            with self.subTest(i=i):
                self.assertEqual(value.keys(), self.order_keys.keys())

    def test_output_shape_for_terms(self):
        for i, value in enumerate([self.term_r, self.term_c, self.term_r_2,
                                   self.term_c_2]):
            with self.subTest(i=i):
                self.assertEqual(value.keys(), self.term_keys.keys())

    def test_basis_shapes_for_orders(self):
        for i, value in enumerate([self.order_r, self.order_c]):
            for n, basis in value.items():
                with self.subTest(i=(i, n)):
                    nb_coeff = kernel_nb_coeff(self.M, n, form='sym')
                    self.assertEqual(basis.shape, (self.L, nb_coeff))

    def test_basis_shapes_for_orders_2(self):
        for i, value in enumerate([self.order_r_2, self.order_c_2]):
            for n, basis in value.items():
                with self.subTest(i=(i, n)):
                    nb_coeff = kernel_nb_coeff(self.M_2[n-1], n, form='sym')
                    self.assertEqual(basis.shape, (self.L, nb_coeff))

    def test_basis_shapes_for_terms(self):
        for i, value in enumerate([self.term_r, self.term_c]):
            for (n, q), basis in value.items():
                with self.subTest(i=(i, (n, q))):
                    nb_coeff = kernel_nb_coeff(self.M, n, form='sym')
                    self.assertEqual(basis.shape, (self.L, nb_coeff))

    def test_basis_shapes_for_terms_2(self):
        for i, value in enumerate([self.term_r_2, self.term_c_2]):
            for (n, q), basis in value.items():
                with self.subTest(i=(i, (n, q))):
                    nb_coeff = kernel_nb_coeff(self.M_2[n-1], n, form='sym')
                    self.assertEqual(basis.shape, (self.L, nb_coeff))

    def test_same_result_with_term_and_order_on_real_signals(self):
        for n in range(1, self.N+1):
            with self.subTest(i=n):
                self.assertTrue(np.all(self.order_r[n] == self.term_r[(n, 0)]))

    def test_same_result_with_term_and_order_on_real_signals_2(self):
        for n in range(1, self.N+1):
            with self.subTest(i=n):
                self.assertTrue(np.all(self.order_r_2[n] ==
                                       self.term_r_2[(n, 0)]))

    def test_same_result_with_term_and_order_on_complex_signals(self):
        for n in range(1, self.N+1):
            with self.subTest(i=n):
                self.assertTrue(np.all(self.order_c[n] == self.term_c[(n, 0)]))

    def test_same_result_with_term_and_order_on_complex_signals_2(self):
        for n in range(1, self.N+1):
            with self.subTest(i=n):
                self.assertTrue(np.all(self.order_c_2[n] ==
                                       self.term_c_2[(n, 0)]))

    def test_same_result_between_all_terms_with_real_signals(self):
        for n in range(1, self.N+1):
            term = self.term_r[(n, 0)]
            for q in range(1, 1+n//2):
                with self.subTest(i=(n, q)):
                    self.assertTrue(np.all(term == self.term_r[(n, q)]))

    def test_same_result_between_all_terms_with_real_signals_2(self):
        for n in range(1, self.N+1):
            term = self.term_r_2[(n, 0)]
            for q in range(1, 1+n//2):
                with self.subTest(i=(n, q)):
                    self.assertTrue(np.all(term == self.term_r_2[(n, q)]))


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
