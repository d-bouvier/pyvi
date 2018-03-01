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
import itertools
import numpy as np
from pyvi.identification.tools import (_as_list, nb_coeff_in_kernel,
                                       nb_coeff_in_all_kernels,
                                       assert_enough_data_samples,
                                       vector_to_kernel, kernel_to_vector,
                                       vector_to_all_kernels, volterra_basis)
from pyvi.utilities.mathbox import binomial


#==============================================================================
# Test Class
#==============================================================================

class AsListTest(unittest.TestCase):

    def test_int_returns_list(self):
        m = 0
        for N in range(1, 4):
            with self.subTest(i=N):
                self.assertIsInstance(_as_list(m, N), list)

    def test_list_returns_list(self):
        M = [1, 2, 3]
        self.assertIsInstance(_as_list(M, len(M)), list)

    def test_error_if_wrong_length(self):
        M = [1, 2, 3]
        self.assertRaises(ValueError, _as_list, M, len(M)-1)

    def test_return_value_correct(self):
        m = 0
        N = 3
        self.assertEqual(_as_list(m, N), [m, ]*N)


class NbCoeffInKernelTest(unittest.TestCase):

    def setUp(self):
        self.Nmax = 5
        self.Mmax = 20
        self.iter_obj = itertools.product(range(1, self.Nmax+1),
                                          range(self.Mmax))

    def test_nb_coeff_symmetric_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = nb_coeff_in_kernel(M, N, form='sym')
                self.assertEqual(nb_coeff, binomial(M + N - 1, N))


    def test_nb_coeff_triangular_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = nb_coeff_in_kernel(M, N, form='tri')
                self.assertEqual(nb_coeff, binomial(M + N - 1, N))

    def test_nb_coeff_raw_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = nb_coeff_in_kernel(M, N, form=None)
                self.assertEqual(nb_coeff, M**N)


class NbCoeffInAllKernelsTest(unittest.TestCase):

    def setUp(self):
        self.Nmax = 5
        self.Mmax = 5
        self.M_list = [10, 0, 3, 0, 2]
        self.iter_obj = itertools.product(range(1, self.Nmax+1),
                                          range(self.Mmax))

    def test_nb_coeff_symmetric_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = nb_coeff_in_all_kernels(M, N, form='sym')
                self.assertEqual(nb_coeff, binomial(M + N, N) - 1)


    def test_nb_coeff_triangular_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = nb_coeff_in_all_kernels(M, N, form='tri')
                self.assertEqual(nb_coeff, binomial(M + N, N) - 1)

    def test_nb_coeff_raw_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = nb_coeff_in_all_kernels(M, N, form=None)
                self.assertEqual(nb_coeff, sum([M**N for N in range(1, N+1)]))

    def test_M_as_list(self):
        for form, val in [('sym', 26), ('tri', 26), (None, 69)]:
            with self.subTest(i=form):
                nb_coeff = nb_coeff_in_all_kernels(self.M_list,
                                                   len(self.M_list), form=form)
                self.assertEqual(nb_coeff, val)


class AssertEnoughDataSamplesTest(unittest.TestCase):

    def test_error_raised(self):
        self.assertRaises(ValueError, assert_enough_data_samples, 8, 9,
                          3, 2, 'KLS')


class VectorToKernelTest(unittest.TestCase):

    def setUp(self):
        self.M = 4
        self.h_vec = {2: np.arange(1, binomial(self.M + 1, 2)+1),
                      3: np.arange(1, binomial(self.M + 2, 3)+1)}
        self.h_tri = {2: np.array([[1, 2, 3, 4],
                                   [0, 5, 6, 7],
                                   [0, 0, 8, 9],
                                   [0, 0, 0, 10]]),
                      3: np.array([[[1, 2, 3, 4],
                                    [0, 5, 6, 7],
                                    [0, 0, 8, 9],
                                    [0, 0, 0, 10]],
                                   [[0, 0, 0, 0],
                                    [0, 11, 12, 13],
                                    [0, 0, 14, 15],
                                    [0, 0, 0, 16]],
                                   [[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 17, 18],
                                    [0, 0, 0, 19]],
                                   [[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 20]]])}
        self.h_sym = {2: np.array([[1, 1, 1.5, 2],
                                   [1, 5, 3, 3.5],
                                   [1.5, 3, 8, 4.5],
                                   [2, 3.5, 4.5, 10]]),
                      3: np.array([[[1., 2/3, 1, 4/3],
                                    [2/3, 5/3, 1, 7/6],
                                    [1, 1, 8/3, 1.5],
                                    [4/3, 7/6, 1.5, 10/3]],
                                   [[2/3, 5/3, 1, 7/6],
                                    [5/3, 11, 4, 13/3],
                                    [1, 4, 14/3, 2.5],
                                    [7/6, 13/3, 2.5, 16/3]],
                                   [[1, 1, 8/3, 1.5],
                                    [1, 4, 14/3, 2.5],
                                    [8/3, 14/3, 17, 6],
                                    [1.5, 2.5, 6, 19/3]],
                                   [[4/3, 7/6, 1.5, 10/3],
                                    [7/6, 13/3, 2.5, 16/3],
                                    [1.5, 2.5, 6, 19/3],
                                    [10/3, 16/3, 19/3, 20]]])}

    def test_triangular_form(self):
        for n in [2, 3]:
            with self.subTest(i=n):
                result = vector_to_kernel(self.h_vec[n], self.M, n, form='tri')
                self.assertTrue(np.all(result == self.h_tri[n]))

    def test_symmetric_form(self):
        for n in [2, 3]:
            with self.subTest(i=n):
                result = vector_to_kernel(self.h_vec[n], self.M, n, form='sym')
                self.assertTrue(np.all(result == self.h_sym[n]))

    def test_error_raised(self):
        n = 2
        self.assertRaises(ValueError, vector_to_kernel, self.h_vec[n], self.M,
                          n+1)


class KernelToVectorTest(VectorToKernelTest):

    def test_triangular_form(self):
        for n in [2, 3]:
            with self.subTest(i=n):
                result = kernel_to_vector(self.h_tri[n], form='tri')
                self.assertTrue(np.all(result == self.h_vec[n]))

    def test_symmetric_form(self):
        for n in [2, 3]:
            with self.subTest(i=n):
                result = kernel_to_vector(self.h_sym[n], form='sym')
                self.assertTrue(np.all(result == self.h_vec[n]))


class VectorToAllKerrnelsTest(VectorToKernelTest):

    def setUp(self):
        VectorToKernelTest.setUp(self)
        self.N = 3
        self.h_vec[1] = np.arange(1, self.M+1)
        self.f = np.concatenate((self.h_vec[1], self.h_vec[2], self.h_vec[3]),
                                axis=0)
        self.f_dict = {1: self.h_vec[1], 2: self.h_vec[2], 3: self.h_vec[3]}
        self.h_tri[1] = self.h_vec[1]
        self.h_sym[1] = self.h_vec[1]

        self.M_2 = [4, 3, 2]
        self.f_2 = np.concatenate((np.arange(1, binomial(self.M_2[0], 1)+1),
                                   np.arange(1, binomial(self.M_2[1]+1, 2)+1),
                                   np.arange(1, binomial(self.M_2[2]+2, 3)+1)),
                                  axis=0)
        self.f_dict_2 = {1: np.arange(1, binomial(self.M_2[0], 1)+1),
                         2: np.arange(1, binomial(self.M_2[1]+1, 2)+1),
                         3: np.arange(1, binomial(self.M_2[2]+2, 3)+1)}
        self.h_tri_2 = {1: np.array([1, 2, 3, 4]),
                        2: np.array([[1, 2, 3],
                                     [0, 4, 5],
                                     [0, 0, 6]]),
                        3: np.array([[[1, 2],
                                      [0, 3]],
                                     [[0, 0],
                                      [0, 4]]])}
        self.h_sym_2 = {1: np.array([1, 2, 3, 4]),
                        2: np.array([[1., 1, 3/2],
                                     [1, 4, 5/2],
                                     [3/2, 5/2, 6]]),
                        3: np.array([[[1., 2/3],
                                      [2/3, 1]],
                                     [[2/3, 1],
                                      [1, 4]]])}

    def test_triangular_form(self):
        kernels = vector_to_all_kernels(self.f, self.M, self.N, form='tri')
        result = [np.all(h == self.h_tri[n]) for n, h in kernels.items()]
        self.assertTrue(all(result))

    def test_symmetric_form(self):
        kernels = vector_to_all_kernels(self.f, self.M, self.N, form='sym')
        result = [np.all(h == self.h_sym[n]) for n, h in kernels.items()]
        self.assertTrue(all(result))

    def test_triangular_form_2(self):
        kernels = vector_to_all_kernels(self.f_2, self.M_2, self.N, form='tri')
        result = [np.all(h == self.h_tri_2[n]) for n, h in kernels.items()]
        self.assertTrue(all(result))

    def test_symmetric_form_2(self):
        kernels = vector_to_all_kernels(self.f_2, self.M_2, self.N, form='sym')
        result = [np.all(h == self.h_sym_2[n]) for n, h in kernels.items()]
        self.assertTrue(all(result))

    def test_error_raised_if_wrong_size(self):
        self.assertRaises(ValueError, vector_to_all_kernels, self.f, self.M,
                          self.N+1)

    def test_error_raised_if_wrong_type(self):
        f = list()
        self.assertRaises(TypeError, vector_to_all_kernels, f, self.M, self.N)

    def test_f_as_dict(self):
        kernels = vector_to_all_kernels(self.f_dict, self.M, self.N,
                                        form='sym')
        result = [np.all(h == self.h_sym[n]) for n, h in kernels.items()]
        self.assertTrue(all(result))

    def test_f_as_dict_2(self):
        kernels = vector_to_all_kernels(self.f_dict_2, self.M_2, self.N,
                                        form='sym')
        result = [np.all(h == self.h_sym_2[n]) for n, h in kernels.items()]
        self.assertTrue(all(result))


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
                    nb_coeff = nb_coeff_in_kernel(self.M, n, form='sym')
                    self.assertEqual(basis.shape, (self.L, nb_coeff))

    def test_basis_shapes_for_orders_2(self):
        for i, value in enumerate([self.order_r_2, self.order_c_2]):
            for n, basis in value.items():
                with self.subTest(i=(i, n)):
                    nb_coeff = nb_coeff_in_kernel(self.M_2[n-1], n, form='sym')
                    self.assertEqual(basis.shape, (self.L, nb_coeff))

    def test_basis_shapes_for_terms(self):
        for i, value in enumerate([self.term_r, self.term_c]):
            for (n, q), basis in value.items():
                with self.subTest(i=(i, (n, q))):
                    nb_coeff = nb_coeff_in_kernel(self.M, n, form='sym')
                    self.assertEqual(basis.shape, (self.L, nb_coeff))

    def test_basis_shapes_for_terms_2(self):
        for i, value in enumerate([self.term_r_2, self.term_c_2]):
            for (n, q), basis in value.items():
                with self.subTest(i=(i, (n, q))):
                    nb_coeff = nb_coeff_in_kernel(self.M_2[n-1], n, form='sym')
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
