# -*- coding: utf-8 -*-
"""
Test script for pyvi/utilities/mathbox.py

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
import pyvi.utilities.mathbox as mathbox


#==============================================================================
# Test Class
#==============================================================================

class RmsTestCase(unittest.TestCase):

    def test_computation(self):
        for i in range(3):
            with self.subTest(i=i):
                array = i * np.ones((10, 1))
                self.assertEqual(mathbox.rms(array), i)

    def test_shape(self):
        array = (2 * np.ones((24, 1))).reshape(2, 3, 4)
        shape = {0: (3, 4), 1: (2, 4), 2: (2, 3)}
        for i in [0, 1, 2]:
            with self.subTest(i=i):
                self.assertEqual(mathbox.rms(array, axis=i).shape, shape[i],
                                 'wrong output shape')


class DbTestCase(unittest.TestCase):

    def test_computation(self):
        for i in range(-5, 6):
            with self.subTest(i=i):
                self.assertEqual(mathbox.db(10**i), 20*i)

    def test_shape(self):
        array = np.array([1, 2, 3])
        with self.subTest(i=0):
            self.assertEqual(mathbox.db(array, array).shape, array.shape,
                             'wrong output shape')
        with self.subTest(i=1):
            self.assertEqual(mathbox.db(array, 1).shape, array.shape,
                             'wrong output shape')

    def test_ref(self):
        for i in range(-5, 6):
            with self.subTest(i=i):
                self.assertEqual(mathbox.db(1., ref=10**i), -20*i)


class SafedbTestCase(unittest.TestCase):

    def test_computation(self):
        for i in range(-5, 6):
            with self.subTest(i=i):
                self.assertEqual(mathbox.safe_db(10**i, 1.), 20*i)

    def test_den_null(self):
        self.assertEqual(mathbox.safe_db(1, 0), np.Inf)

    def test_num_null(self):
        self.assertEqual(mathbox.safe_db(0, 1), - np.Inf)

    def test_den_and_num_null(self):
        self.assertEqual(mathbox.safe_db(0, 0), - np.Inf)

    def test_output_type(self):
        array = np.ones((2))
        tests_map = {'int_int': (1, 1, float),
                     'float_float': (1., 1., float),
                     'float_int': (1., 1, float),
                     'int_float': (1, 1., float),
                     'list_list': ([1, 1], [1, 1], np.ndarray),
                     'list_array': ([1, 1], array, np.ndarray),
                     'array_list': (array, [1, 1], np.ndarray),
                     'array_array': (array, array, np.ndarray)}
        for name, (num, den, out_type) in tests_map.items():
            with self.subTest(name=name):
                self.assertIsInstance(mathbox.safe_db(num, den), out_type)

    def test_wrong_shape_error(self):
        tests_map = {'list3_list2': ([1, 1, 1], [1, 1]),
                     'num3_num2': (np.ones((3, 1)), np.ones((2, 1)))}
        for name, (num, den) in tests_map.items():
            with self.subTest(name=name):
                self.assertRaises(ValueError, mathbox.safe_db, num, den)


class BinomialTestCase(unittest.TestCase):

    def test_n_choose_0(self):
        for n in range(1, 10):
            with self.subTest(i=n):
                self.assertEqual(mathbox.binomial(n, 0), 1)

    def test_n_choose_1(self):
        for n in range(1, 10):
            with self.subTest(i=n):
                self.assertEqual(mathbox.binomial(n, 1), n)

    def test_n_choose_n(self):
        for n in range(1, 10):
            with self.subTest(i=n):
                self.assertEqual(mathbox.binomial(n, n), 1)

    def test_symmetry(self):
        for n in range(1, 10):
            for k in range(1, n):
                with self.subTest(name='({}, {})'.format(n, k)):
                    self.assertEqual(mathbox.binomial(n, k),
                                     mathbox.binomial(n, n-k))

    def test_recursivity(self):
        for n in range(2, 10):
            for k in range(1, n):
                with self.subTest(name='({}, {})'.format(n, k)):
                    self.assertEqual(mathbox.binomial(n, k),
                                     mathbox.binomial(n-1, k-1) +
                                     mathbox.binomial(n-1, k))


#TODO: class MultinomialTestCase(unittest.TestCase):


class ArraySymmetrizationTestCase(unittest.TestCase):

    def test_symmetrization(self):
        array = np.array([[1, 2, 4],
                          [0, 3, 6],
                          [0, 0, 8]])
        array_sym = np.array([[1, 1, 2],
                              [1, 3, 3],
                              [2, 3, 8]])
        array_sym_est = mathbox.array_symmetrization(array)
        self.assertTrue(np.all(array_sym == array_sym_est))


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
