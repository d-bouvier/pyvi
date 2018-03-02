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
from pyvi.identification.tools import assert_enough_data_samples


#==============================================================================
# Test Class
#==============================================================================

class AssertEnoughDataSamplesTest(unittest.TestCase):

    def test_error_raised(self):
        self.assertRaises(ValueError, assert_enough_data_samples, 8, 9,
                          3, 2, 'KLS')


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
