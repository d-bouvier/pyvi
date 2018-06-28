# -*- coding: utf-8 -*-
"""
Test script for pyvi/utilities/tools.py

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
from pyvi.utilities.tools import _as_list, _is_sorted, inherit_docstring


#==============================================================================
# Test Class
#==============================================================================

class A():

    def foo(self):
        """Docstring of A.foo"""
        pass


class B(A):

    @inherit_docstring
    def foo(self):
        pass


class C(A):

    def foo(self):
        """Docstring of C.foo"""
        pass

    def bar(self):
        """Docstring of C.bar"""
        pass


class D(B, C):
    pass


class E(D):

    @inherit_docstring
    def foo(self):
        pass

    @inherit_docstring
    def bar(self):
        pass


class InheritDocstringTest(unittest.TestCase):

    list_foo = [(A, B), (A, D), (A, E)]
    list_bar = [(C, D), (C, E)]

    def test_foo_correct_doc_without_instance(self):
        for (cls1, cls2) in self.list_foo:
            with self.subTest(i=(cls1, cls2)):
                self.assertEqual(cls1.foo.__doc__, cls2.foo.__doc__)

    def test_bar_correct_doc_without_instance(self):
        for (cls1, cls2) in self.list_bar:
            with self.subTest(i=(cls1, cls2)):
                self.assertEqual(cls1.bar.__doc__, cls2.bar.__doc__)

    def test_foo_correct_doc_with_instances(self):
        for (cls1, cls2) in self.list_foo:
            with self.subTest(i=(cls1, cls2)):
                obj1 = cls1()
                obj2 = cls2()
                self.assertEqual(obj1.foo.__doc__, obj2.foo.__doc__)

    def test_bar_correct_doc_with_instances(self):
        for (cls1, cls2) in self.list_bar:
            with self.subTest(i=(cls1, cls2)):
                obj1 = cls1()
                obj2 = cls2()
                self.assertEqual(obj1.bar.__doc__, obj2.bar.__doc__)


class AsListTest(unittest.TestCase):

    def test_any_returns_list(self):
        val = 0
        for N in range(1, 4):
            with self.subTest(i=N):
                self.assertIsInstance(_as_list(val, N), list)

    def test_return_value_correct(self):
        val = 0
        for N in range(1, 4):
            with self.subTest(i=N):
                self.assertEqual(len(set(_as_list(val, N))), 1)

    def test_list_returns_list(self):
        val = [1, 2, 3]
        self.assertIsInstance(_as_list(val, len(val)), list)

    def test_tuple_returns_list(self):
        val = (1, 2, 3)
        self.assertIsInstance(_as_list(val, len(val)), list)

    def test_numpy_array_returns_list(self):
        val = np.array([1, 2, 3])
        self.assertIsInstance(_as_list(val, len(val)), list)

    def test_str_returns_list_of_str(self):
        val = 'test'
        N = 4
        should_be_result = list((val,)*N)
        self.assertListEqual(_as_list(val, N), should_be_result)

    def test_error_if_wrong_length(self):
        val = [1, 2, 3]
        self.assertRaises(ValueError, _as_list, val, len(val)-1)


class IsSortedTest(unittest.TestCase):

    arrays = [([1, 2, 3], True),
              ([1, 3, 2], False),
              ([1, 3, 3], True)]

    def test_correct(self):
        for (array, result) in self.arrays:
            with self.subTest(i=array):
                self.assertEqual(_is_sorted(array), result)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
