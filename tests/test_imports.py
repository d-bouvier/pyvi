# -*- coding: utf-8 -*-
"""
Test script for checking correctness of the namespaces and their attributes.

Notes
-----
Developed for Python 3.6
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import unittest
import pyvi


#==============================================================================
# Test Class
#==============================================================================

class PyviTestCase(unittest.TestCase):

    module = pyvi
    needed_properties = ['__author__', '__maintainer__', '__version__',
                         'utilities', 'volterra', 'separation',
                         'identification']

    def test_should_be_present_properties(self):
        for property_name in self.needed_properties:
            with self.subTest(i=property_name):
                self.assertTrue(hasattr(self.module, property_name))


class UtilitiesTestCase(PyviTestCase):

    module = pyvi.utilities
    needed_properties = ['rms', 'db', 'safe_db', 'binomial', 'multinomial',
                         'array_symmetrization', 'separation_error',
                         'identification_error', 'LaguerreBasis', 'KautzBasis',
                         'GeneralizedBasis', 'create_orthogonal_basis',
                         'is_valid_basis_instance']


class VolterraTestCase(PyviTestCase):

    module = pyvi.volterra
    needed_properties = ['kernel_nb_coeff', 'series_nb_coeff', 'vec2kernel',
                         'vec2series', 'kernel2vec', 'volterra_basis',
                         'hammerstein_basis', 'projected_volterra_basis',
                         'projected_hammerstein_basis']


class SeparationTestCase(PyviTestCase):

    module = pyvi.separation
    needed_properties = ['AS', 'CPS', 'HPS', 'RPS', 'PAS']


class IdentificationTestCase(PyviTestCase):

    module = pyvi.identification
    needed_properties = ['direct_method', 'order_method', 'term_method',
                         'iter_method', 'phase_method',
                         'compute_combinatorial_basis']


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()

