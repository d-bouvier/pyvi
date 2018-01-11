# -*- coding: utf-8 -*-
"""
Test script for pyvi/separation/methods.py

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import unittest
import warnings
import numpy as np
import pyvi.separation.methods as sep


#==============================================================================
# Test Class
#==============================================================================

class _SeparationMethodGlobalTest():

    method_class = None
    input_dtype = 'float'
    signal_dtype = 'float'
    atol = 1e-12
    N = 4

    def setUp(self):
        self.L = 2000
        if self.input_dtype == 'float':
            input_sig = np.random.normal(size=(self.L,))
        else:
            input_sig = np.random.normal(size=(self.L,)) + \
                        1j * np.random.normal(size=(self.L,))
        self.method = self.method_class(N=self.N)
        input_coll = self.method.gen_inputs(input_sig)
        self.output_coll = np.zeros(input_coll.shape, dtype=self.signal_dtype)
        for ind in range(input_coll.shape[0]):
            self.output_coll[ind] = generate_output(input_coll[ind], self.N)
        self.order_est = self.method.process_outputs(self.output_coll)
        if (self.input_dtype == 'complex') & (self.signal_dtype == 'float'):
            input_sig = 2 * np.real(input_sig)
        self.order_true = generate_output(input_sig, self.N, by_order=True)

    def test_shape(self):
        self.assertEqual(self.order_est.shape, (self.N, self.L))

    def test_correct_output(self):
        self.assertTrue(np.allclose(self.order_est, self.order_true,
                                    rtol=0, atol=self.atol))


class ASMethodTestCase(_SeparationMethodGlobalTest, unittest.TestCase):

    method_class = sep.AS


class _PSMethodTestCase(_SeparationMethodGlobalTest, unittest.TestCase):

    method_class = sep._PS
    input_dtype = 'complex'
    signal_dtype = 'complex'


class PSMethodTestCase(_SeparationMethodGlobalTest, unittest.TestCase):

    method_class = sep.PS
    input_dtype = 'complex'
    signal_dtype = 'float'
    N = 2

    def setUp(self):
        _SeparationMethodGlobalTest.setUp(self)
        save = self.order_est
        self.order_est = np.stack((save[1] + save[4],
                                   save[0] + save[2] + save[3]), axis=0)
        self.order_est = np.real(self.order_est)


class PASMethodTestCase(_SeparationMethodGlobalTest, unittest.TestCase):

    method_class = sep.PAS
    input_dtype = 'complex'
    signal_dtype = 'float'
    atol = 1e-10

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            _SeparationMethodGlobalTest.setUp(self)

    def test_shape_term(self):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            _, term_est = self.method.process_outputs(self.output_coll,
                                                      raw_mode=True)
        keys = dict()
        for n in range(1, self.N+1):
            for k in range(n+1):
                keys[(n, k)] = ()
        self.assertEqual(term_est.keys(), keys.keys())


class PASv2MethodTestCase(PASMethodTestCase, unittest.TestCase):

    method_class = sep.PAS_v2
    input_dtype = 'complex'
    signal_dtype = 'float'
    atol = 1e-10


#==============================================================================
# Functions
#==============================================================================

def generate_output(input_sig, N, by_order=False):
    output_by_order = np.zeros((N, len(input_sig)), dtype=input_sig.dtype)
    for n in range(N):
        output_by_order[n, :] = input_sig**(n+1)
        output_by_order[n, 1:] += input_sig[:-1]**(n+1)
        output_by_order[n, 1:] -= 2*input_sig[:-1]**n * input_sig[1:]
    if by_order:
        return output_by_order
    else:
        return np.sum(output_by_order, axis=0)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
