# -*- coding: utf-8 -*-
"""
Test script for pyvi/system/statespace.py

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 24 nov. 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import unittest
import warnings
import numpy as np
from pyvi.system.statespace import (StateSpace, NumericalStateSpace)


#==============================================================================
# Test Class
#==============================================================================

class StateSpaceTestCase(unittest.TestCase):

    def test_check_dim_mat(self):
        mat_shapes = [[(3, 3, 3), (3, 2), (2, 3), (2, 2)],
                      [(3, 4), (3, 2), (2, 3), (2, 2)],
                      [(3, 3), (4, 2), (2, 3), (2, 2)],
                      [(3, 3), (4), (2, 3), (2, 1)],
                      [(3, 3), (3, 2), (2, 4), (2, 2)],
                      [(3, 3), (3, 2), (4), (1, 2)],
                      [(3, 3), (3, 2), (2, 3), (2, 4)],
                      [(3, 3), (3,), (2, 3), (2, 2)],
                      [(3, 3), (3, 2), (2, 3), (2,)],
                      [(3, 3), (3, 2), (2, 3), (4, 2)],
                      [(3, 3), (3,), (2, 3), (2, 2)],
                      [(3, 3), (3, 2), (2, 3), (2)]]
        for i, (shapeA, shapeB, shapeC, shapeD) in enumerate(mat_shapes):
            with self.subTest(i=i):
                self.assertRaises(AssertionError, StateSpace,
                                  np.empty(shapeA), np.empty(shapeB),
                                  np.empty(shapeC), np.empty(shapeD))

    def test_check_dim_pq(self):
        shapeA = (3, 3)
        shapeB = (3, 2)
        shapeC = (2, 3)
        shapeD = (2, 2)
        pq_val = [[{(2, 0): np.empty((3, 3, 4))}, {}],
                  [{(2, 0): np.empty((4, 3, 3))}, {}],
                  [{(1, 1): np.empty((3, 3, 1))}, {}],
                  [{(1, 1): np.empty((3, 4, 2))}, {}],
                  [{(1, 1): np.empty((4, 3, 2))}, {}],
                  [{(0, 2): np.empty((3, 2, 1))}, {}],
                  [{(0, 2): np.empty((4, 2, 2))}, {}],
                  [{}, {(2, 0): np.empty((2, 3, 4))}],
                  [{}, {(2, 0): np.empty((3, 3, 3))}],
                  [{}, {(1, 1): np.empty((2, 3, 1))}],
                  [{}, {(1, 1): np.empty((2, 4, 2))}],
                  [{}, {(1, 1): np.empty((3, 3, 2))}],
                  [{}, {(0, 2): np.empty((2, 2, 1))}],
                  [{}, {(0, 2): np.empty((3, 2, 2))}]]
        for i, (mpq, npq) in enumerate(pq_val):
            with self.subTest(i=i):
                self.assertRaises(AssertionError, StateSpace,
                                  np.empty(shapeA), np.empty(shapeB),
                                  np.empty(shapeC), np.empty(shapeD),
                                  mpq=mpq, npq=npq)

    def test_warning_if_non_siso(self):
        dim = [[2, 3, 1],
               [1, 3, 2],
               [2, 3, 2]]
        for i, (dim_in, dim_state, dim_out) in enumerate(dim):
            with self.subTest(i=i):
                self.assertWarns(UserWarning, StateSpace,
                                 np.empty((dim_state, dim_state)),
                                 np.empty((dim_state, dim_in)),
                                 np.empty((dim_out, dim_state)),
                                 np.empty((dim_out, dim_in)))

    def test_attribute_type(self):
        tests = {'SISO': [1, 3, 1],
                 'MISO': [2, 3, 1],
                 'SIMO': [1, 3, 2],
                 'MIMO': [2, 3, 2]}
        warnings.filterwarnings("ignore")
        for type_sys, (dim_in, dim_state, dim_out) in tests.items():
            with self.subTest(i=type_sys):
                system = StateSpace(np.empty((dim_state, dim_state)),
                                    np.empty((dim_state, dim_in)),
                                    np.empty((dim_out, dim_state)),
                                    np.empty((dim_out, dim_in)))
                self.assertEqual(system._type, type_sys)
        warnings.filterwarnings("default")

    def test_attribute_linear(self):
        n_in = 1
        n_s = 3
        n_out = 1
        A = np.empty((n_s, n_s))
        B = np.empty((n_s, n_in))
        C = np.empty((n_out, n_s))
        D = np.empty((n_out, n_in))
        tests = [[True, {}, {}],
                 [False, {(2, 0): np.empty((n_s, n_s, n_s))}, {}],
                 [False, {}, {(2, 0): np.empty((n_out, n_s, n_s))}]]
        for i, (result, mpq, npq) in enumerate(tests):
            with self.subTest(i=i):
                system = StateSpace(A, B, C, D, mpq=mpq, npq=npq)
                self.assertEqual(system.linear, result)


class StateSpaceDynamicalAttributesTestCase(unittest.TestCase):

    def setUp(self):
        n_in = 1
        n_s = 3
        n_out = 1
        self.A = np.empty((n_s, n_s))
        self.B = np.empty((n_s, n_in))
        self.C = np.empty((n_out, n_s))
        self.D = np.empty((n_out, n_in))
        self.m20 = np.empty((n_s, n_s, n_s))
        self.m11 = np.empty((n_s, n_s, n_in))
        self.m02 = np.empty((n_s, n_in, n_in))
        self.tests = [{}, {(2, 0): self.m20}, {(1, 1): self.m11},
                      {(0, 2): self.m02}]

    def test_attribute_input_affine(self):
        results = [True, True, True, False]
        for i, mpq in enumerate(self.tests):
            with self.subTest(i=mpq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, mpq=mpq)
                self.assertEqual(system._dyn_eqn_input_affine, results[i])

    def test_attribute_state_affine(self):
        results = [True, False, True, True]
        for i, mpq in enumerate(self.tests):
            with self.subTest(i=mpq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, mpq=mpq)
                self.assertEqual(system._dyn_eqn_state_affine, results[i])

    def test_attribute_nl_only_on_input(self):
        results = [True, False, False, True]
        for i, mpq in enumerate(self.tests):
            with self.subTest(i=mpq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, mpq=mpq)
                self.assertEqual(system._dyn_nl_only_on_input, results[i])

    def test_attribute_nl_only_on_state(self):
        results = [True, True, False, False]
        for i, mpq in enumerate(self.tests):
            with self.subTest(i=mpq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, mpq=mpq)
                self.assertEqual(system._dyn_nl_only_on_state, results[i])

    def test_attribute_eqn_linear(self):
        results = [True, False, False, False]
        for i, mpq in enumerate(self.tests):
            with self.subTest(i=mpq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, mpq=mpq)
                self.assertEqual(system._dyn_eqn_linear, results[i])

    def test_attribute_eqn_bilinear(self):
        results = [True, False, True, False]
        for i, mpq in enumerate(self.tests):
            with self.subTest(i=mpq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, mpq=mpq)
                self.assertEqual(system._dyn_eqn_bilinear, results[i])


class StateSpaceOutputAttributesTestCase(unittest.TestCase):

    def setUp(self):
        n_in = 1
        n_s = 3
        n_out = 1
        self.A = np.empty((n_s, n_s))
        self.B = np.empty((n_s, n_in))
        self.C = np.empty((n_out, n_s))
        self.D = np.empty((n_out, n_in))
        self.n20 = np.empty((n_out, n_s, n_s))
        self.n11 = np.empty((n_out, n_s, n_in))
        self.n02 = np.empty((n_out, n_in, n_in))
        self.tests = [{}, {(2, 0): self.n20}, {(1, 1): self.n11},
                      {(0, 2): self.n02}]

    def test_attribute_input_affine(self):
        results = [True, True, True, False]
        for i, npq in enumerate(self.tests):
            with self.subTest(i=npq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, npq=npq)
                self.assertEqual(system._out_eqn_input_affine, results[i])

    def test_attribute_state_affine(self):
        results = [True, False, True, True]
        for i, npq in enumerate(self.tests):
            with self.subTest(i=npq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, npq=npq)
                self.assertEqual(system._out_eqn_state_affine, results[i])

    def test_attribute_nl_only_on_input(self):
        results = [True, False, False, True]
        for i, npq in enumerate(self.tests):
            with self.subTest(i=npq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, npq=npq)
                self.assertEqual(system._out_nl_only_on_input, results[i])

    def test_attribute_nl_only_on_state(self):
        results = [True, True, False, False]
        for i, npq in enumerate(self.tests):
            with self.subTest(i=npq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, npq=npq)
                self.assertEqual(system._out_nl_only_on_state, results[i])

    def test_attribute_eqn_linear(self):
        results = [True, False, False, False]
        for i, npq in enumerate(self.tests):
            with self.subTest(i=npq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, npq=npq)
                self.assertEqual(system._out_eqn_linear, results[i])

    def test_attribute_eqn_bilinear(self):
        results = [True, False, True, False]
        for i, npq in enumerate(self.tests):
            with self.subTest(i=npq.keys()):
                system = StateSpace(self.A, self.B, self.C, self.D, npq=npq)
                self.assertEqual(system._out_eqn_bilinear, results[i])


class NumericalStateSpaceTestCase(unittest.TestCase):

    def test_attribute_dynamical_nl_only_on_state(self):
        n_in = 1
        n_s = 3
        n_out = 1
        A = np.empty((n_s, n_s))
        B = np.array([[1], [0], [0]])
        C = np.empty((n_out, n_s))
        D = np.empty((n_out, n_in))
        m20a = np.zeros((n_s, n_s, n_s))
        m20d = np.zeros((n_s, n_s, n_s))
        m20a[1, 1, 2] = 10
        m20a[1, 2, 1] = 10
        m20a[1, 1, 1] = 5
        m20b = m20a.copy()
        m20b[1] = 2
        m20c = m20b.copy()
        m20c[2, 1, 1] = 5
        m20d = np.zeros((n_s, n_s, n_s))
        m20d[0] = 2
        tests = [[False, B, {(2, 0): m20a}],
                 [False, B, {(2, 0): m20b}],
                 [False, B, {(2, 0): m20c}],
                 [True, B, {(2, 0): m20d}]]
        for i, (result, B, mpq) in enumerate(tests):
            with self.subTest(i=i):
                system = NumericalStateSpace(A, B, C, D, mpq=mpq)
                self.assertEqual(system._nl_colinear, result)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
