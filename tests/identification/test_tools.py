# -*- coding: utf-8 -*-
"""
Test script for pyvi/identification/tools.py

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
from pyvi.identification.tools import (_solver, _complex2real,
                                       _check_parameters,
                                       _compute_list_nb_coeff)
from pyvi.utilities.orthogonal_basis import LaguerreBasis


#==============================================================================
# Test Class
#==============================================================================

class SolverTest(unittest.TestCase):

    atol = 1e-15

    def setUp(self):
        self.A = np.array([[1., 0.5],
                           [0.33, 2.]])
        self.x = np.ones((2,))
        self.y = np.dot(self.A, self.x)
        self.list_solvers = ['LS', 'ls', 'QR', 'qr']

    def test_correct_output(self):
        for solver in self.list_solvers:
            with self.subTest(i=solver):
                x_est = _solver(self.A, self.y, solver)
                result = np.allclose(self.x, x_est, atol=self.atol, rtol=0)
                self.assertTrue(result)

    def test_A_empty(self):
        self.assertEqual(_solver(np.zeros((0,)), self.y, 'ls').size, 0)

    def test_wrong_solver(self):
        self.assertRaises(ValueError, _solver, self.A, self.y, '')


class Complex2RealTest(unittest.TestCase):

    def setUp(self):
        self.val = np.array([1 + 2j, 3 + 4j])
        self.real = np.array([1, 3])
        self.imag = np.array([2, 4])
        self.real_imag = np.array([1, 3, 2, 4])

    def test_default_mode(self):
        result = _complex2real(self.val)
        self.assertTrue(np.all(result == self.real_imag))

    def test_real_mode(self):
        result = _complex2real(self.val, cast_mode='real')
        self.assertTrue(np.all(result == self.real))

    def test_imag_mode(self):
        result = _complex2real(self.val, cast_mode='imag')
        self.assertTrue(np.all(result == self.imag))

    def test_real_imag_mode(self):
        result = _complex2real(self.val, cast_mode='real-imag')
        self.assertTrue(np.all(result == self.real_imag))

    def test_cplx_mode(self):
        result = _complex2real(self.val, cast_mode='cplx')
        self.assertTrue(np.all(result == self.val))

    def test_warns(self):
        self.assertWarns(UserWarning, _complex2real, self.val, cast_mode='')


class CheckParametersTest(unittest.TestCase):

    N = 3
    basis = LaguerreBasis(0.1, 3)

    def test_wrong_system_type_error(self):
        self.assertRaises(ValueError, _check_parameters, self.N, '', 5, None)

    def test_no_M_and_no_basis_error(self):
        self.assertRaises(ValueError, _check_parameters, self.N, 'volterra',
                          None, None)

    def test_n_M_and_basis_error(self):
        self.assertWarns(UserWarning, _check_parameters, self.N, 'volterra',
                         5, self.basis)

    def test_wrong_M_error(self):
        for M in (0.5, [0.5, 2, 3]):
            with self.subTest(i=M):
                self.assertRaises(TypeError, _check_parameters, self.N,
                                  'volterra', M, None)

    def test_wrong_basis_error(self):
        for basis in (0.5, [self.basis, self.basis, dict()]):
            with self.subTest(i=basis):
                self.assertRaises(TypeError, _check_parameters, self.N,
                                  'volterra', None, basis)

    def test_is_list_false(self):
        M, orthogonal_basis_is_list = _check_parameters(self.N, 'volterra',
                                                        None, self.basis)
        self.assertFalse(orthogonal_basis_is_list)

    def test_is_list_true(self):
        M, orthogonal_basis_is_list = _check_parameters(self.N, 'volterra',
                                                        None, (self.basis,)*3)
        self.assertTrue(orthogonal_basis_is_list)


class ComputeListNbCoeffParametersTest(unittest.TestCase):

    N = 3
    M = 3
    M_list = [3, 2, 3]
    basis = LaguerreBasis(0.1, M)
    basis_list = [LaguerreBasis(0.1, m) for m in M_list]
    result = {'volterra': [3, 6, 10], 'hammerstein': list([M, ]*N)}
    result_list = {'volterra': [3, 3, 10], 'hammerstein': M_list}


    def test_proj(self):
        for basis, is_list, result in [(self.basis, False, self.result),
                                       (self.basis_list, True,
                                        self.result_list)]:
            for sys_type in {'volterra', 'hammerstein'}:
                with self.subTest(i=(sys_type, basis)):
                    nb_coeff = _compute_list_nb_coeff(self.N, sys_type, None,
                                                      basis, is_list)
                    self.assertEqual(nb_coeff, result[sys_type])

    def test_no_proj(self):
        for M, is_list, result in [(self.M, False, self.result),
                                   (self.M_list, False, self.result_list)]:
            for sys_type in {'volterra', 'hammerstein'}:
                with self.subTest(i=(sys_type, M)):
                    nb_coeff = _compute_list_nb_coeff(self.N, sys_type, M,
                                                      None, is_list)
                    self.assertEqual(nb_coeff, result[sys_type])


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
