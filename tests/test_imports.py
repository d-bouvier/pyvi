# -*- coding: utf-8 -*-
"""
Test script for checking correctness of the namespaces and their attributes.

Notes
-----
Developed for Python 3.6.1
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
    should_be_absent_properties = []

    def test_should_be_present_properties(self):
        for property_name in self.needed_properties:
            with self.subTest(i=property_name):
                self.assertTrue(hasattr(self.module, property_name))

    def test_should_be_absent_properties(self):
        for property_name in self.should_be_absent_properties:
            with self.subTest(i=property_name):
                self.assertFalse(hasattr(self.module, property_name))


class UtilitiesTestCase(PyviTestCase):

    module = pyvi.utilities
    needed_properties = ['rms', 'db', 'safe_db', 'binomial', 'multinomial',
                         'array_symmetrization', 'separation_error',
                         'identification_error', 'LaguerreBasis', 'KautzBasis',
                         'GeneralizedBasis', 'create_orthogonal_basis',
                         'is_valid_basis_instance']
    should_be_absent_properties = ['_AbstractOrthogonalBasis',
                                   'inherint_docstring', '_as_list']


class VolterraTestCase(PyviTestCase):

    module = pyvi.volterra
    needed_properties = ['kernel_nb_coeff', 'series_nb_coeff', 'vec2kernel',
                         'vec2series', 'kernel2vec',
                         'compute_combinatorial_basis', 'volterra_basis',
                         'hammerstein_basis', 'projected_volterra_basis',
                         'projected_hammerstein_basis']
    should_be_absent_properties = ['_vec2dict_of_vec', '_check_parameters',
                                   '_compute_list_nb_coeff',
                                   '_phi_by_order_post_processing',
                                   '_combinatorial_mat_diag_terms']


class SeparationTestCase(PyviTestCase):

    module = pyvi.separation
    needed_properties = ['AS', 'CPS', 'HPS', 'PS', 'PAS']
    should_be_absent_properties = ['_SeparationMethod', '_AbstractPS',
                                   '_create_vandermonde_mixing_mat']



class IdentificationTestCase(PyviTestCase):

    module = pyvi.identification
    needed_properties = ['direct_method', 'order_method', 'term_method',
                         'iter_method', 'phase_method', 'KLS', 'orderKLS',
                         'termKLS', 'iterKLS', 'phaseKLS']
    should_be_absent_properties = ['_solver', '_ls_solver', '_qr_solver',
                                   '_complex2real', '_identification',
                                   '_cast_complex2real', '_kwargs_for_KLS']


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()

