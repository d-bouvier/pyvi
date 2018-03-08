# -*- coding: utf-8 -*-
"""
Test script for pyvi/identification/methods.py

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import itertools as itr
import unittest
import numpy as np
from pyvi.identification.methods import (direct_method, order_method,
                                         term_method, iter_method,
                                         phase_method)
from pyvi.separation.methods import HPS, PS
from pyvi.volterra.tools import kernel2vec
from pyvi.volterra.combinatorial_basis import volterra_basis
from pyvi.utilities.mathbox import array_symmetrization
from pyvi.utilities.tools import _as_list


#==============================================================================
# Test Class
#==============================================================================

class DirectMethodTest(unittest.TestCase):

    N = 4
    M = 3
    L = 100
    rtol = 0
    atol = 1e-12
    method = staticmethod(direct_method)
    solvers = {'LS', 'QR'}
    cast_modes = {'real', 'imag', 'real-imag'}

    def _create_input(self):
        self.input_sig = np.random.normal(size=(self.L,))

    def _create_output(self):
        self.output_data = generate_output(self.input_sig, self.kernels_vec,
                                           self.N, self.M)

    def _identification(self):
        self.list_kernels_est = dict()
        for solver, cast_mode in itr.product(self.solvers, self.cast_modes):
            self.list_kernels_est[(solver, cast_mode)] = \
                self.method(self.input_sig, self.output_data, self.N,
                            solver=solver, cast_mode=cast_mode, **self.kwargs)

    def setUp(self):
        self.kwargs = {'M': self.M, 'out_form': 'sym'}
        self.kernels_true, self.kernels_vec = generate_kernels(self.N, self.M)
        self._M = _as_list(self.M, self.N)
        self._create_input()
        self._create_output()
        self._identification()

    def test_check_keys_dict(self):
        keys = set(range(1, self.N+1))
        for key, kernels_est in self.list_kernels_est.items():
            with self.subTest(i=key):
                self.assertSetEqual(set(kernels_est.keys()), keys)

    def test_check_shape_kernels(self):
        for key, kernels_est in self.list_kernels_est.items():
            for n, h in kernels_est.items():
                with self.subTest(i=(n, key)):
                    self.assertEqual(h.shape, (self._M[n-1],)*n)

    def test_correct_output(self):
        for key, kernels_est in self.list_kernels_est.items():
            for n, h in kernels_est.items():
                with self.subTest(i=(n, key)):
                    self.assertTrue(np.allclose(h, self.kernels_true[n],
                                                rtol=self.rtol,
                                                atol=self.atol))


class OrderMethodTest(DirectMethodTest):

    method = staticmethod(order_method)

    def _create_output(self):
        self.output_data = generate_output(self.input_sig, self.kernels_vec,
                                           self.N, self.M, by_order=True)


class TermMethodTest(DirectMethodTest):

    method = staticmethod(term_method)

    def _create_input(self):
        self.input_sig = np.random.normal(size=(self.L,)) + \
                         1j * np.random.normal(size=(self.L,))

    def _create_output(self):
        sep_method = PS(self.N)
        input_coll = sep_method.gen_inputs(self.input_sig)
        output_coll = np.zeros(input_coll.shape)
        for ind in range(input_coll.shape[0]):
            output_coll[ind] = generate_output(input_coll[ind],
                                               self.kernels_vec, self.N,
                                               self.M)
        _, self.output_data = sep_method.process_outputs(output_coll,
                                                         raw_mode=True)


class IterMethodTest(TermMethodTest):

    method = staticmethod(iter_method)

    def _create_output(self):
        sep_method = HPS(self.N)
        input_coll = sep_method.gen_inputs(self.input_sig)
        output_coll = np.zeros(input_coll.shape)
        for ind in range(input_coll.shape[0]):
            output_coll[ind] = generate_output(input_coll[ind],
                                               self.kernels_vec, self.N,
                                               self.M)
        self.output_data = sep_method.process_outputs(output_coll)


class PhaseMethodTest(IterMethodTest):

    method = staticmethod(phase_method)


class DirectMethod_ListM_Test(DirectMethodTest):

    M = [3, 5, 0, 5]


class OrderMethod_ListM_Test(OrderMethodTest, DirectMethod_ListM_Test):

    pass


class TermMethod_ListM_Test(TermMethodTest, DirectMethod_ListM_Test):

    pass


class IterMethod_ListM_Test(IterMethodTest, DirectMethod_ListM_Test):

    pass


class PhaseMethod_ListM_Test(PhaseMethodTest, DirectMethod_ListM_Test):

    pass


#==============================================================================
# Functions
#==============================================================================

def generate_output(input_sig, kernels_vec, N, M, by_order=False):
    phi = volterra_basis(input_sig, N, M, sorted_by='order')
    L = phi[1].shape[0]
    output_by_order = np.zeros((N, L))
    for n in range(N):
        output_by_order[n, :] = np.dot(phi[n+1], kernels_vec[n+1])
    if by_order:
        return output_by_order
    else:
        return np.sum(output_by_order, axis=0)


def generate_kernels(N, M):
    _M = _as_list(M, N)
    kernels = dict()
    kernels_vec = dict()
    for m, n in zip(_M, range(N)):
        temp = np.zeros((m,)*(n+1))
        if m:
            temp[(0,)*(n+1)] = 1
            temp[(m-1,)*(n+1)] = 1
        if m and n:
            temp[(0,) + (m-1,)*n] = -2
        kernels[n+1] = array_symmetrization(temp)
        kernels_vec[n+1] = kernel2vec(kernels[n+1], form='sym')
    return kernels, kernels_vec


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
