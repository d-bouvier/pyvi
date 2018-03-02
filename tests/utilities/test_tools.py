# -*- coding: utf-8 -*-
"""
Test script for pyvi/utilities/tools.py

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import unittest
from pyvi.utilities.tools import _as_list


#==============================================================================
# Test Class
#==============================================================================

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

    def test_error_if_wrong_length(self):
        val = [1, 2, 3]
        self.assertRaises(ValueError, _as_list, val, len(val)-1)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
