# -*- coding: utf-8 -*-
"""
Test script for pyvi/volterra/combinatorial_basis.py

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
from pyvi.volterra.combinatorial_basis import (volterra_basis,
                                               hammerstein_basis,
                                               projected_volterra_basis,
                                               projected_hammerstein_basis)
from pyvi.volterra.tools import kernel_nb_coeff
from pyvi.utilities.orthogonal_basis import LaguerreBasis
from pyvi.utilities.tools import _as_list


#==============================================================================
# Test Class
#==============================================================================

class _CombinatorialBasisTest():

    L = 30
    N = 4
    order_keys = {1, 2, 3, 4}
    term_keys = {(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1),
                 (4, 2)}
    M = 15

    def setUp(self):
        sig = self.sig_creation()
        self._M = _as_list(self.M, self.N)
        self.order = self.compute_basis_func(sig, sorted_by='order')
        self.term = self.compute_basis_func(sig, sorted_by='term')
        self.nb_coeff = dict()
        for n in range(1, self.N+1):
            self.nb_coeff[n] = self.compute_nb_coeff(n)

    def compute_basis_func(self, sig, sorted_by):
        return None

    def compute_nb_coeff(self, n):
        return kernel_nb_coeff(n, self._M[n-1], form='tri')

    def sig_creation(self):
        return np.arange(1, self.L+1)

    def test_output_type(self):
        for i, value in enumerate([self.order, self.term]):
            with self.subTest(i=i):
                self.assertIsInstance(value, dict)

    def test_output_keys_for_orders(self):
        self.assertSetEqual(set(self.order.keys()), self.order_keys)

    def test_output_keys_for_terms(self):
        self.assertSetEqual(set(self.term.keys()), self.term_keys)

    def test_basis_shapes_for_orders(self):
        for n, basis in self.order.items():
            with self.subTest(i=n):
                self.assertEqual(basis.shape, (self.L, self.nb_coeff[n]))

    def test_basis_shapes_for_terms(self):
        for (n, k), basis in self.term.items():
            with self.subTest(i=(n, k)):
                self.assertEqual(basis.shape, (self.L, self.nb_coeff[n]))

    def test_same_result_with_term_and_order(self):
        for n in range(1, self.N+1):
            with self.subTest(i=n):
                self.assertTrue(np.all(self.order[n] == self.term[(n, 0)]))

    def test_same_result_between_all_terms_with_real_signals(self):
        for n in range(1, self.N+1):
            term = self.term[(n, 0)]
            for k in range(1, 1+n//2):
                with self.subTest(i=(n, k)):
                    self.assertTrue(np.all(term == self.term[(n, k)]))


class VolterraBasisTest(_CombinatorialBasisTest, unittest.TestCase):

    def compute_basis_func(self, sig, sorted_by):
        return volterra_basis(sig, self.N, self.M, sorted_by=sorted_by)


class VolterraBasis_M_List_Test(VolterraBasisTest):

    M = [10, 15, 0, 5]


class VolterraBasisCplxSigTest(VolterraBasis_M_List_Test):

    test_same_result_between_all_terms_with_real_signals = property()

    def sig_creation(self):
        return super().sig_creation() + 1j * super().sig_creation()


class VolterraBasisCorrectOutputTest(VolterraBasisCplxSigTest):

    L = 4
    N = 3
    M = 3
    true = {(1, 0): np.array([[1+1j, 2+2j, 3+3j, 4+4j],
                              [  0., 1+1j, 2+2j, 3+3j],
                              [  0.,   0., 1+1j, 2+2j]]).T,
            (2, 0): np.array([[2j, 8j, 18j, 32j],
                              [0., 4j, 12j, 24j],
                              [0.,  0,  6j, 16j],
                              [0., 2j,  8j, 18j],
                              [0., 0.,  4j, 12j],
                              [0., 0.,  2j,  8j]]).T,
            (2, 1): np.array([[2., 8., 18., 32.],
                              [0., 4., 12., 24.],
                              [0.,  0,  6., 16.],
                              [0., 2.,  8., 18.],
                              [0., 0.,  4., 12.],
                              [0., 0.,  2.,  8.]]).T,
            (3, 0): np.array([[-2+2j, -16+16j, -54+54j, -128+128j],
                              [   0.,   -8+8j, -36+36j,   -96+96j],
                              [   0.,      0., -18+18j,   -64+64j],
                              [   0.,   -4+4j, -24+24j,   -72+72j],
                              [   0.,      0., -12+12j,   -48+48j],
                              [   0.,      0.,   -6+6j,   -32+32j],
                              [   0.,   -2+2j, -16+16j,   -54+54j],
                              [   0.,      0.,   -8+8j,   -36+36j],
                              [   0.,      0.,   -4+4j,   -24+24j],
                              [   0.,      0.,   -2+2j,   -16+16j]]).T,
            (3, 1): np.array([[2+2j, 16+16j, 54+54j, 128+128j],
                              [  0.,   8+8j, 36+36j,   96+96j],
                              [  0.,     0., 18+18j,   64+64j],
                              [  0.,   4+4j, 24+24j,   72+72j],
                              [  0.,     0., 12+12j,   48+48j],
                              [  0.,     0.,   6+6j,   32+32j],
                              [  0.,   2+2j, 16+16j,   54+54j],
                              [  0.,     0.,   8+8j,   36+36j],
                              [  0.,     0.,   4+4j,   24+24j],
                              [  0.,     0.,   2+2j,   16+16j]]).T}
    order_keys = {1, 2, 3}
    term_keys = {(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)}

    def test_correct_output_order(self):
        for n, val in self.order.items():
            with self.subTest(i=n):
                self.assertTrue(np.all(val == self.true[(n, 0)]))

    def test_correct_output_term(self):
        for (n, k), val in self.term.items():
            with self.subTest(i=(n, k)):
                self.assertTrue(np.all(val == self.true[(n, k)]))


class HammersteinBasisTest(VolterraBasisTest):

    def compute_basis_func(self, sig, sorted_by):
        return hammerstein_basis(sig, self.N, self.M, sorted_by=sorted_by)

    def compute_nb_coeff(self, n):
        return self._M[n-1]


class HammersteinBasis_M_List_Test(HammersteinBasisTest,
                                   VolterraBasis_M_List_Test):
    pass


class HammersteinBasisCplxSigTest(HammersteinBasis_M_List_Test,
                                  VolterraBasisCplxSigTest):
    true = {(1, 0): np.array([[1+1j, 2+2j, 3+3j, 4+4j],
                              [  0., 1+1j, 2+2j, 3+3j],
                              [  0.,   0., 1+1j, 2+2j]]).T,
            (2, 0): np.array([[2j, 8j, 18j, 32j],
                              [0., 2j,  8j, 18j],
                              [0., 0.,  2j,  8j]]).T,
            (2, 1): np.array([[2., 8., 18., 32.],
                              [0., 2.,  8., 18.],
                              [0., 0.,  2.,  8.]]).T,
            (3, 0): np.array([[-2+2j, -16+16j, -54+54j, -128+128j],
                              [   0.,   -2+2j, -16+16j,   -54+54j],
                              [   0.,      0.,   -2+2j,   -16+16j]]).T,
            (3, 1): np.array([[2+2j, 16+16j, 54+54j, 128+128j],
                              [  0.,   2+2j, 16+16j,   54+54j],
                              [  0.,     0.,   2+2j,   16+16j]]).T}
    pass


class HammersteinBasisCorrectOutputTest(HammersteinBasisCplxSigTest,
                                        VolterraBasisCorrectOutputTest):
    pass


class ProjectedVolterraBasisTest(VolterraBasisTest):

    L = 200
    pole = 0.1
    M = 5
    rtol = 0
    atol = 1e-12

    def compute_basis_func(self, sig, sorted_by):
        if isinstance(self.M, int):
            orthogonal_basis = LaguerreBasis(self.pole, self.M)
        else:
            orthogonal_basis = [LaguerreBasis(self.pole, m) for m in self.M]
        return projected_volterra_basis(sig, self.N, orthogonal_basis,
                                        sorted_by=sorted_by)

    def sig_creation(self):
        return np.random.normal(size=(self.L,))

    def test_same_result_between_all_terms_with_real_signals(self):
        for n in range(1, self.N+1):
            term = self.term[(n, 0)]
            for k in range(1, 1+n//2):
                with self.subTest(i=(n, k)):
                    self.assertTrue(np.allclose(term, self.term[(n, k)],
                                                rtol=self.rtol,
                                                atol=self.atol))


class ProjectedVolterraBasisListTest(ProjectedVolterraBasisTest):

    M = [3, 5, 0, 5]


class ProjectedVolterraBasisCplxSigTest(ProjectedVolterraBasisListTest):

    test_same_result_between_all_terms_with_real_signals = property()

    def sig_creation(self):
        return super().sig_creation() + 1j * super().sig_creation()


class ProjectedVolterraBasisCorrectOutputTest(ProjectedVolterraBasisTest):

    L = 3
    N = 3
    M = 3
    atol = 1e-7
    rtol = 1e-12
    true = {(1, 0): np.array(
                [[0.+0.j, 0.99498744+0.99498744j, 2.08947362+2.08947362j],
                 [0.+0.j, 0.09949874+0.09949874j, -0.7760902-0.7760902j],
                 [0.+0.j, 0.00994987+0.00994987j, -0.17611278-0.17611278j]]).T,
            (2, 0): np.array(
                [[0.+0.000000e+00j, 0.+1.980000e+00j, 0.+8.731800e+00j],
                 [0.+0.j, 0.+0.198j, 0.-3.24324j],
                 [0.+0.j, 0.+0.0198j, 0.-0.735966j],
                 [0.+0.j, 0.+1.980000e-02j, 0.+1.204632e+00j],
                 [0.+0.j, 0.+0.00198j, 0.+0.2733588j],
                 [0.+0.j, 0.+1.980000e-04j, 0.+6.203142e-02j]]).T,
            (2, 1): np.array(
                 [[0., 1.980000, 8.731800],
                  [0., 0.198, -3.24324],
                  [0., 0.0198, -0.735966],
                  [0., 1.980000e-02, 1.204632],
                  [0., 0.00198, 0.2733588],
                  [0., 1.980000e-04, 6.203142e-02]]).T,
            (3, 0): np.array(
                 [[0.+0.j, -1.97007513+1.97007513j, -18.2448657+18.2448657j],
                  [0.+0.j, -0.19700751+0.19700751j, 6.77666442-6.77666442j],
                  [0.+0.j, -0.01970075+0.01970075j, 1.53778154-1.53778154j],
                  [0.+0.j, -1.97007513e-02+1.97007513e-02j,
                   -2.51704678+2.51704678j],
                  [0.+0.j, -1.97007513e-03+1.97007513e-03j,
                   -5.71176001e-01+5.71176001e-01j],
                  [0.+0.j, -0.00019701+0.00019701j, -0.12961302+0.12961302j],
                  [0.+0.j, -1.97007513e-03+1.97007513e-03j,
                   9.34903091e-01-9.34903091e-01j],
                  [0.+0.j, -1.97007513e-04+1.97007513e-04j,
                   2.12151086e-01-2.12151086e-01j],
                  [0.+0.j, -1.97007513e-05+1.97007513e-05j,
                   4.81419772e-02-4.81419772e-02j],
                  [0.+0.j, -1.97007513e-06+1.97007513e-06j,
                   1.09245256e-02-1.09245256e-02j]]).T,
            (3, 1): np.array(
                  [[0.+0.j, 1.97007513e+00+1.97007513e+00j,
                    1.82448657e+01+1.82448657e+01j],
                   [0.+0.j, 0.19700751+0.19700751j, -6.77666442-6.77666442j],
                   [0.+0.j, 0.01970075+0.01970075j, -1.53778154-1.53778154j],
                   [0.+0.j, 0.01970075+0.01970075j, 2.51704678+2.51704678j],
                   [0.+0.j, 0.00197008+0.00197008j, 0.571176+0.571176j],
                   [0.+0.j, 0.00019701+0.00019701j, 0.12961302+0.12961302j],
                   [0.+0.j, 1.97007513e-03+1.97007513e-03j,
                    -9.34903091e-01-9.34903091e-01j],
                   [0.+0.j, 1.97007513e-04+1.97007513e-04j,
                    -2.12151086e-01-2.12151086e-01j],
                   [0.+0.j, 1.97007513e-05+1.97007513e-05j,
                    -4.81419772e-02-4.81419772e-02j],
                   [0.+0.j, 1.97007513e-06+1.97007513e-06j,
                    -1.09245256e-02-1.09245256e-02j]]).T}
    order_keys = {1, 2, 3}
    term_keys = {(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)}

    test_same_result_between_all_terms_with_real_signals = property()

    def sig_creation(self):
        return np.arange(1, self.L+1) + 1j * np.arange(1, self.L+1)

    def test_correct_output_order(self):
        for n, val in self.order.items():
            with self.subTest(i=n):
                self.assertTrue(np.allclose(val, self.true[(n, 0)],
                                            rtol=self.rtol, atol=self.atol))

    def test_correct_output_term(self):
        for (n, k), val in self.term.items():
            with self.subTest(i=(n, k)):
                self.assertTrue(np.allclose(val, self.true[(n, k)],
                                            rtol=self.rtol, atol=self.atol))


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
