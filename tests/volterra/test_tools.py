# -*- coding: utf-8 -*-
"""
Test script for pyvi/volterra/tools.py

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
from pyvi.volterra.tools import (kernel_nb_coeff, series_nb_coeff, vec2kernel,
                                 vec2dict_of_vec, vec2series, kernel2vec)
from pyvi.utilities.mathbox import binomial


#==============================================================================
# Test Class
#==============================================================================

class KernelNbCoeffTest(unittest.TestCase):

    def setUp(self):
        self.Nmax = 5
        self.Mmax = 20
        self.iter_obj = itertools.product(range(1, self.Nmax+1),
                                          range(self.Mmax))

    def test_nb_coeff_symmetric_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = kernel_nb_coeff(N, M, form='sym')
                self.assertEqual(nb_coeff, binomial(M + N - 1, N))


    def test_nb_coeff_triangular_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = kernel_nb_coeff(N, M, form='tri')
                self.assertEqual(nb_coeff, binomial(M + N - 1, N))

    def test_nb_coeff_raw_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = kernel_nb_coeff(N, M, form=None)
                self.assertEqual(nb_coeff, M**N)


class SeriesNbCoeffTest(unittest.TestCase):

    def setUp(self):
        self.Nmax = 5
        self.Mmax = 5
        self.M_list = [10, 0, 3, 0, 2]
        self.M_list_results = [('sym', 26), ('tri', 26), (None, 69)]
        self.form_list = [None, None, 'sym', 'tri', 'sym']
        self.iter_obj = itertools.product(range(1, self.Nmax+1),
                                          range(self.Mmax))

    def test_nb_coeff_symmetric_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = series_nb_coeff(N, M, form='sym')
                self.assertEqual(nb_coeff, binomial(M + N, N) - 1)


    def test_nb_coeff_triangular_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = series_nb_coeff(N, M, form='tri')
                self.assertEqual(nb_coeff, binomial(M + N, N) - 1)

    def test_nb_coeff_raw_form(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = series_nb_coeff(N, M, form=None)
                self.assertEqual(nb_coeff, sum([M**n for n in range(1, N+1)]))

    def test_form_as_list(self):
        M = self.Mmax
        N = self.Nmax
        val = binomial(M + N, N) - 1 + binomial(M, 2)
        nb_coeff = series_nb_coeff(N, M, form=self.form_list)
        self.assertEqual(nb_coeff, val)

    def test_M_as_list(self):
        for form, val in self.M_list_results:
            with self.subTest(i=form):
                nb_coeff = series_nb_coeff(len(self.M_list), self.M_list,
                                           form=form)
                self.assertEqual(nb_coeff, val)

    def test_out_by_order_type(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = series_nb_coeff(N, M, out_by_order=True)
                self.assertIsInstance(nb_coeff, list)

    def test_out_by_order_length(self):
        for N, M in self.iter_obj:
            with self.subTest(i=(N, M)):
                nb_coeff = series_nb_coeff(N, M, out_by_order=True)
                self.assertEqual(len(nb_coeff), N)


class Vec2KernelTest(unittest.TestCase):

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
                result = vec2kernel(self.h_vec[n], n, self.M, form='tri')
                self.assertTrue(np.all(result == self.h_tri[n]))

    def test_symmetric_form(self):
        for n in [2, 3]:
            with self.subTest(i=n):
                result = vec2kernel(self.h_vec[n], n, self.M, form='sym')
                self.assertTrue(np.all(result == self.h_sym[n]))

    def test_None_form(self):
        for n in [2, 3]:
            with self.subTest(i=n):
                result = vec2kernel(self.h_vec[n], n, self.M, form=None)
                self.assertTrue(np.all(result == self.h_tri[n]))

    def test_error_raised(self):
        n = 2
        self.assertRaises(ValueError, vec2kernel, self.h_vec[n], n+1, self.M)


class Vec2SeriesErrorTest(unittest.TestCase):

    def test_error_raised_if_wrong_type(self):
        f = list()
        self.assertRaises(TypeError, vec2series, f, 3, 3)


class Vec2SeriesTest(Vec2KernelTest):

    test_error_raised = property()

    def setUp(self):
        super().setUp()
        self.N = 3
        self.h_vec[1] = np.arange(1, self.M+1)
        self.f = {1: self.h_vec[1], 2: self.h_vec[2], 3: self.h_vec[3]}
        self.h_tri[1] = self.h_vec[1]
        self.h_sym[1] = self.h_vec[1]

    def test_triangular_form(self):
        kernels = vec2series(self.h_vec, self.N, self.M, form='tri')
        result = [np.all(h == self.h_tri[n]) for n, h in kernels.items()]
        self.assertTrue(all(result))

    def test_symmetric_form(self):
        kernels = vec2series(self.h_vec, self.N, self.M, form='sym')
        result = [np.all(h == self.h_sym[n]) for n, h in kernels.items()]
        self.assertTrue(all(result))

    def test_None_form(self):
        kernels = vec2series(self.h_vec, self.N, self.M, form=None)
        result = [np.all(h == self.h_tri[n]) for n, h in kernels.items()]
        self.assertTrue(all(result))


class Vec2Series_F_AsVector_Test(Vec2SeriesTest):

    def setUp(self):
        super().setUp()
        self.h_vec = np.concatenate([f for n, f in sorted(self.h_vec.items())],
                                    axis=0)


class Vec2Series_M_AsList_Test(Vec2SeriesTest):

    def setUp(self):
        super().setUp()
        self.M = [4, 3, 2]
        self.h_vec = {1: np.arange(1, binomial(self.M[0], 1)+1),
                      2: np.arange(1, binomial(self.M[1]+1, 2)+1),
                      3: np.arange(1, binomial(self.M[2]+2, 3)+1)}
        self.h_tri = {1: np.array([1, 2, 3, 4]),
                      2: np.array([[1, 2, 3],
                                   [0, 4, 5],
                                   [0, 0, 6]]),
                      3: np.array([[[1, 2],
                                    [0, 3]],
                                   [[0, 0],
                                    [0, 4]]])}
        self.h_sym = {1: np.array([1, 2, 3, 4]),
                      2: np.array([[1., 1, 3/2],
                                   [1, 4, 5/2],
                                   [3/2, 5/2, 6]]),
                      3: np.array([[[1., 2/3],
                                    [2/3, 1]],
                                   [[2/3, 1],
                                    [1, 4]]])}


class Vec2Series_Form_AsList_Test(Vec2KernelTest):

    test_triangular_form = property()
    test_symmetric_form = property()
    test_None_form = property()

    def setUp(self):
        super().setUp()
        self.N = 3
        self.form = ['sym', 'tri', None]
        self.h_vec[1] = np.arange(1, self.M+1)
        self.h = dict()
        self.h[1] = self.h_vec[1]
        self.h[2] = self.h_tri[2]
        self.h[3] = self.h_tri[3]

    def test_f_as_dict(self):
        kernels = vec2series(self.h_vec, self.N, self.M, form=self.form)
        result = [np.all(h == self.h[n]) for n, h in kernels.items()]
        self.assertTrue(all(result))


class Vec2DictOfVec(Vec2SeriesTest):

    test_triangular_form = property()
    test_symmetric_form = property()
    test_None_form = property()
    test_error_raised = property()

    def setUp(self):
        super().setUp()
        h_vec = np.concatenate([f for n, f in sorted(self.h_vec.items())],
                               axis=0)
        self.out = vec2dict_of_vec(h_vec, self.N, self.M)

    def test_output_dict(self):
        self.assertIsInstance(self.out, dict)

    def test_correct_keys(self):
        self.assertListEqual(list(self.out.keys()), list(range(1, self.N+1)))

    def test_correct_output(self):
        result = [np.all(h == self.h_vec[n]) for n, h in self.out.items()]
        self.assertTrue(all(result))


class Kernel2VecTest(Vec2KernelTest):

    def setUp(self):
        super().setUp()
        self.h_raw = {2: np.array([[1, 1, 3, 4],
                                   [1, 5, 3, 7],
                                   [0, 3, 8, 9],
                                   [0, 0, 0, 10]]),
                      3: np.array([[[1, 2, 1, 4],
                                    [0, 5, 6, 7],
                                    [1, 0, 8, 9],
                                    [0, 0, 0, 10]],
                                   [[0, 0, 0, 0],
                                    [0, 11, 12, 13],
                                    [0, 0, 14, 15],
                                    [0, 0, 0, 16]],
                                   [[1, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 17, 18],
                                    [0, 0, 0, 19]],
                                   [[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 20]]])}

    def test_triangular_form(self):
        for n in [2, 3]:
            with self.subTest(i=n):
                result = kernel2vec(self.h_tri[n], form='tri')
                self.assertTrue(np.all(result == self.h_vec[n]))

    def test_symmetric_form(self):
        for n in [2, 3]:
            with self.subTest(i=n):
                result = kernel2vec(self.h_sym[n], form='sym')
                self.assertTrue(np.all(result == self.h_vec[n]))

    def test_None_form(self):
        for n in [2, 3]:
            with self.subTest(i=n):
                result = kernel2vec(self.h_raw[n], form=None)
                self.assertTrue(np.all(result == self.h_vec[n]))

    def test_error_raised(self):
        h_not_squared = np.zeros((3, 3, 4))
        self.assertRaises(ValueError, kernel2vec, h_not_squared)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
