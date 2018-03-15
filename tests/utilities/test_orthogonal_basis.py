# -*- coding: utf-8 -*-
"""
Test script for pyvi/utilities/orthogonal_basis.py

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
from pyvi.utilities.orthogonal_basis import (_OrthogonalBasis, LaguerreBasis,
                                             KautzBasis, GeneralizedBasis,
                                             create_orthogonal_basis,
                                             is_valid_basis_instance)


#==============================================================================
# Test Class
#==============================================================================

class CreateOrthogonalBasisTest(unittest.TestCase):

    K = 6

    def test_real_pole_outputs_laguerre(self):
        for pole in [0.5, 0.5 + 0j, [0.1], [0.1 + 0j]]:
            with self.subTest(i=pole):
                self.assertIsInstance(create_orthogonal_basis(pole, self.K),
                                      LaguerreBasis)

    def test_cplx_pole_outputs_kautz(self):
        for pole in [0.5 + 0.1j, [0.5 + 0.1j]]:
            with self.subTest(i=pole):
                self.assertIsInstance(create_orthogonal_basis(pole, self.K),
                                      KautzBasis)

    def test_list_poles_outputs_generalized(self):
        for poles in [[0.5, 0.4], [0.5 + 0.1j, 0.4 + 0.5j], [0.5, 0.5 + 0.1j]]:
            with self.subTest(i=poles):
                self.assertIsInstance(create_orthogonal_basis(poles),
                                      GeneralizedBasis)

    def test_zero_len_error(self):
        self.assertRaises(ValueError, create_orthogonal_basis, [])

    def test_no_K_for_laguerre_error(self):
        self.assertRaises(ValueError, create_orthogonal_basis, 0.5)

    def test_no_K_for_kautz_error(self):
        self.assertRaises(ValueError, create_orthogonal_basis, 0.5 + 0.1j)

    def test_wrong_type_error(self):
        self.assertRaises(TypeError, create_orthogonal_basis, None)


class _OrthogonalBasisGlobalTest():

    params_list = []
    basis = _OrthogonalBasis
    L = 2000
    atol = 1e-14
    rtol = 1e-14

    def setUp(self):
        self.basis_list = []
        for pole, K in self.params_list:
            self.basis_list.append(self.basis(pole, K))

    def test_orthogonality(self):
        input_sig = np.zeros((self.L,))
        input_sig[0] = 1
        for ind, basis in enumerate(self.basis_list):
            with self.subTest(i=ind):
                filters = basis.projection(input_sig)
                orthogonality_mat = np.dot(filters, filters.T)
                self.assertTrue(np.allclose(orthogonality_mat,
                                            np.identity(basis.K),
                                            rtol=self.rtol, atol=self.atol))

    def test_is_valid_basis_obj(self):
        for ind, basis in enumerate(self.basis_list):
            with self.subTest(i=ind):
                self.assertTrue(is_valid_basis_instance(basis))

    def test_cplx_input(self):
        input_real = np.random.normal(size=(100,))
        input_imag = np.random.normal(size=(100,))
        input_sig = input_real + 1j * input_imag
        for ind, basis in enumerate(self.basis_list):
            with self.subTest(i=ind):
                proj_real = basis.projection(input_real)
                proj_imag = basis.projection(input_imag)
                proj_sig = basis.projection(input_sig)
                self.assertTrue(np.allclose(proj_real + 1j * proj_imag,
                                            proj_sig, rtol=self.rtol,
                                            atol=self.atol))


class LaguerreBasisTest(_OrthogonalBasisGlobalTest, unittest.TestCase):
    params_list = [(0.1, 2), (0.1, 5), (0.1, 10), (0.2, 5), (0.5, 5),
                   (0.9, 5), (0.95, 5)]
    basis = LaguerreBasis


class KautzBasisTest(_OrthogonalBasisGlobalTest, unittest.TestCase):
    params_list = [(0.1*np.exp(1j*np.pi/4), 2), (0.1*np.exp(1j*np.pi/4), 10),
                   (0.5*np.exp(1j*np.pi/4), 10), (0.9*np.exp(1j*np.pi/4), 10),
                   (0.95*np.exp(1j*np.pi/4), 10), (0.7 + 0.1j, 10), (0.7, 10)]
    basis = KautzBasis


class GeneralizedBasisTest(_OrthogonalBasisGlobalTest, unittest.TestCase):
    params_list = [[0.1], [0.1, 0.2, 0.5], [0.1, 0.9, 0.1, 0.5, 0.1],
                   [0.1*np.exp(1j*np.pi/4)], [0.1*np.exp(1j*np.pi/4), 0.1],
                   [0.1 + 0.1j, 0.2 + 0.2j, 0.5 + 0.1j, 0.1 + 0.5j],
                   [0.1 + 0.1j, 0.2, 0.5 + 0.1j, 0.5, 0.1 + 0.5j]]
    basis = GeneralizedBasis

    def setUp(self):
        self.basis_list = []
        for poles in self.params_list:
            self.basis_list.append(self.basis(poles))


class RaisedErrorTest(unittest.TestCase):

    def test_cplx_laguerre_pole(self):
        self.assertRaises(ValueError, LaguerreBasis, 1+1j, 2)

    def test_odd_K_for_kautz_basis(self):
        self.assertRaises(ValueError, KautzBasis, 1+1j, 3)


class LaguerreAndGeneralizedEqualityTest(unittest.TestCase):

    pole = 0.5
    list_K = [1, 2, 5, 10]
    comp_basis = LaguerreBasis
    L = 1000
    atol = 1e-14
    rtol = 0

    def setUp(self):
        input_sig = np.zeros((self.L,))
        input_sig[0] = 1
        self.comp_filters = dict()
        self.gob_filters = dict()
        for K in self.list_K:
            poles = self._pole2poles(K)
            comp_basis = self.comp_basis(self.pole, K)
            self.comp_filters[K] = comp_basis.projection(input_sig)
            generalized_basis = GeneralizedBasis(poles)
            self.gob_filters[K] = generalized_basis.projection(input_sig)

    def _pole2poles(self, K):
        return (self.pole,)*K

    def test_orthogonality(self):
        for K in self.list_K:
            with self.subTest(i=K):
                self.assertTrue(np.allclose(self.gob_filters[K],
                                            self.comp_filters[K],
                                            rtol=self.rtol, atol=self.atol))


class KautzAndGeneralizedEqualityTest(LaguerreAndGeneralizedEqualityTest):
    pole = 0.5 + 0.1j
    list_K = [2, 4, 10, 20]
    comp_basis = KautzBasis

    def _pole2poles(self, K):
        return (self.pole,)*(K//2)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
