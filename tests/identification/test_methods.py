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

import unittest
import numpy as np
import pyvi.identification.methods as identif
import pyvi.separation.methods as sep
from pyvi.utilities.mathbox import array_symmetrization
from pyvi.identification.tools import _as_list


#==============================================================================
# Test Class
#==============================================================================

class KLSTest(unittest.TestCase):

    def _init_parameters(self):
        self.M = 2
        self.N = 3
        self.L = 50
        self.rtol = 0
        self.atol = 1e-14

    def _create_input(self):
        self.input_sig = np.random.normal(size=(self.L,))

    def _create_output(self):
        self.output_sig = generate_output(self.input_sig, self.N, self.M)

    def _generate_true_kernels(self):
        self.kernels_true = generate_kernels(self.N, self.M)

    def _identification(self):
        self.kernels_est = identif.KLS(self.input_sig, self.output_sig, self.M,
                                       self.N)

    def setUp(self):
        self._init_parameters()
        self._create_input()
        self._create_output()
        self._generate_true_kernels()
        self._identification()

    def test_check_keys_dict(self):
        keys = {}
        for n in range(1, self.N+1):
            keys[n] = 0
        self.assertEqual(self.kernels_est.keys(), keys.keys())

    def test_check_shape_kernels(self):
        for n, h in self.kernels_est.items():
            with self.subTest(i=n):
                self.assertEqual(h.shape, (self.M,)*n)

    def test_correct_output(self):
        for n, h in self.kernels_est.items():
            with self.subTest(i=n):
                self.assertTrue(np.allclose(h, self.kernels_true[n],
                                            rtol=self.rtol, atol=self.atol))


class KLSTest_M_as_list(KLSTest):

    def _init_parameters(self):
        super()._init_parameters()
        self.M = [2, 0, 3]

    def test_check_shape_kernels(self):
        for m, n in zip(self.M, range(1, self.N+1)):
            with self.subTest(i=n):
                self.assertEqual(self.kernels_est[n].shape, (m,)*n)


class orderKLSTest(KLSTest):

    def _create_output(self):
        self.output_sig_by_order = generate_output(self.input_sig, self.N,
                                                   self.M, by_order=True)

    def _identification(self):
        self.kernels_est = identif.orderKLS(self.input_sig,
                                            self.output_sig_by_order,
                                            self.M, self.N)


class orderKLSTest_M_as_list(orderKLSTest):

    def _init_parameters(self):
        super()._init_parameters()
        self.M = [2, 0, 3]

    def test_check_shape_kernels(self):
        for m, n in zip(self.M, range(1, self.N+1)):
            with self.subTest(i=n):
                self.assertEqual(self.kernels_est[n].shape, (m,)*n)


class termKLSTest(KLSTest):

    def _init_parameters(self):
        KLSTest._init_parameters(self)
        self.atol = 1e-12

    def _create_input(self):
        self.input_sig = np.random.normal(size=(self.L,)) + \
                         1j * np.random.normal(size=(self.L,))

    def _create_output(self):
        method = sep.PAS(N=self.N)
        input_coll = method.gen_inputs(self.input_sig)
        output_coll = np.zeros(input_coll.shape, dtype='complex')
        for ind in range(input_coll.shape[0]):
            output_coll[ind] = generate_output(input_coll[ind], self.N, self.M)
        _, self.output_sig_by_term = method.process_outputs(output_coll,
                                                            raw_mode=True)

    def _identification(self):
        self.kernels_est = identif.termKLS(self.input_sig,
                                           self.output_sig_by_term,
                                           self.M, self.N)


class termKLSTest_M_as_list(termKLSTest):

    def _init_parameters(self):
        super()._init_parameters()
        self.M = [2, 0, 3]

    def test_check_shape_kernels(self):
        for m, n in zip(self.M, range(1, self.N+1)):
            with self.subTest(i=n):
                self.assertEqual(self.kernels_est[n].shape, (m,)*n)


class iterKLSTest(termKLSTest):

    def _init_parameters(self):
        termKLSTest._init_parameters(self)
        self.atol = 1e-12

    def _create_output(self):
        method = sep.HPS(N=self.N)
        input_coll = method.gen_inputs(self.input_sig)
        output_coll = np.zeros(input_coll.shape, dtype='complex')
        for ind in range(input_coll.shape[0]):
            output_coll[ind] = generate_output(input_coll[ind], self.N, self.M)
        self.output_sig_by_phase = method.process_outputs(output_coll)

    def _identification(self):
        self.kernels_est = identif.iterKLS(self.input_sig,
                                           self.output_sig_by_phase,
                                           self.M, self.N)


class iterKLSTest_M_as_list(iterKLSTest):

    def _init_parameters(self):
        super()._init_parameters()
        self.M = [2, 0, 3]

    def test_check_shape_kernels(self):
        for m, n in zip(self.M, range(1, self.N+1)):
            with self.subTest(i=n):
                self.assertEqual(self.kernels_est[n].shape, (m,)*n)


#==============================================================================
# Functions
#==============================================================================

def generate_output(input_sig, N, M, by_order=False):
    M = _as_list(M, N)
    output_by_order = np.zeros((N, len(input_sig)), dtype=input_sig.dtype)
    for m, n in zip(M, range(N)):
        if m:
            output_by_order[n, :] = input_sig**(n+1)
            output_by_order[n, m-1:] += input_sig[:1-m]**(n+1)
        if m and n:
            output_by_order[n, m-1:] -= 2*input_sig[:1-m]**n * input_sig[m-1:]
    if by_order:
        return output_by_order
    else:
        return np.sum(output_by_order, axis=0)


def generate_kernels(N, M):
    M = _as_list(M, N)
    kernels = dict()
    for m, n in zip(M, range(N)):
        temp = np.zeros((m,)*(n+1))
        if m:
            temp[(0,)*(n+1)] = 1
            temp[(m-1,)*(n+1)] = 1
        if m and n:
            temp[(0,) + (m-1,)*n] = -2
        kernels[n+1] = array_symmetrization(temp)
    return kernels


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
