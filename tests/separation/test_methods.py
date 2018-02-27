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
import numpy as np
import pyvi.separation.methods as sep
from pyvi.utilities import binomial, rms


#==============================================================================
# Test Class
#==============================================================================

class _OrderSeparationMethodGlobalTest():

    method = dict()
    input_dtype = {'AS': 'float',
                   'CPS': 'complex',
                   'PS': 'complex',
                   'PAS': 'complex'}
    signal_dtype = {'AS': 'float',
                    'CPS': 'complex',
                    'PS': 'float',
                    'PAS': 'float'}
    true_input_func = {'AS': lambda x: x,
                       'CPS': lambda x: x,
                       'PS': lambda x: 2 * np.real(x),
                       'PAS': lambda x: 2 * np.real(x)}
    tol = 5e-10
    N = 5
    L = 10000

    def setUp(self, **kwargs):
        self.order_est = dict()
        self.order_true = dict()
        for name, method_class in self.method.items():
            if self.input_dtype[name] == 'float':
                input_sig = np.random.normal(size=(self.L,))
            elif self.input_dtype[name] == 'complex':
                input_sig = np.random.normal(size=(self.L,)) + \
                            1j * np.random.normal(size=(self.L,))
            method = method_class(self.N, **kwargs)
            input_coll = method.gen_inputs(input_sig)
            output_coll = np.zeros(input_coll.shape,
                                   dtype=self.signal_dtype[name])
            for ind in range(input_coll.shape[0]):
                output_coll[ind] = generate_output(input_coll[ind], self.N)
            else:
                self.order_est[name] = method.process_outputs(output_coll)
            input_sig = self.true_input_func[name](input_sig)
            self.order_true[name] = generate_output(input_sig, self.N,
                                                    by_order=True)

    def test_shape(self):
        for name in self.method:
            with self.subTest(i=name):
                self.assertEqual(self.order_est[name].shape, (self.N, self.L))

    def test_correct_output(self):
        for name in self.method:
            with self.subTest(i=name):
                error = rms(self.order_est[name] - self.order_true[name])
                self.assertTrue(error < self.tol)


class NoKwargsTestCase(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    method = {'AS': sep.AS,
              'CPS': sep.CPS,
              'PS': sep.PS,
              'PAS': sep.PAS}


class GainTestCase(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    method = {'AS': sep.AS,
              'PAS': sep.PAS}
    tol = 1e-10

    def setUp(self):
        super().setUp(gain=1.5)


class NegativeGainTestCase(_OrderSeparationMethodGlobalTest,
                           unittest.TestCase):

    method = {'AS': sep.AS}

    def setUp(self):
        super().setUp(negative_gain=False)


class K_ArgumentTestCase(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    method = {'AS': sep.AS}

    def setUp(self):
        super().setUp(K=3*self.N)


class NbPhaseTestCase(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    method = {'CPS': sep.CPS,
              'PS': sep.PS,
              'PAS': sep.PAS}
    tol = 5e-10

    def setUp(self):
        super().setUp(nb_phase=32)


class RhoTestCase(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    method = {'CPS': sep.CPS}
    atol = {'CPS': 5e-10}

    def setUp(self):
        super().setUp(rho=2.)


class PS_RawModeTestCase(unittest.TestCase):

    method = sep.PS

    def setUp(self):
        self.N = 3
        self.L = 10000
        method = self.method(self.N)
        output_coll = np.zeros((method.K, self.L))
        _, self.term_est = method.process_outputs(output_coll, raw_mode=True)

    def test_keys(self):
        keys = dict()
        for n in range(1, self.N+1):
            for k in range(n//2 + 1):
                keys[(n, k)] = ()
        self.assertEqual(self.term_est.keys(), keys.keys())


class PAS_RawModeTestCase(PS_RawModeTestCase):

    method = sep.PAS


class HPS_Test(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    method = sep.HPS
    tol = 1e-14

    def setUp(self, **kwargs):
        phase_vec = 2 * np.pi * np.arange(self.L)/self.L
        input_sig = np.exp(1j * phase_vec)
        power_vec = np.arange(1, self.N+1)

        method = self.method(self.N, **kwargs)
        input_coll = method.gen_inputs(input_sig)
        output_coll = np.zeros(input_coll.shape, dtype='complex')
        for ind in range(input_coll.shape[0]):
            tmp = input_coll[ind][np.newaxis, :]**power_vec[:, np.newaxis]
            output_coll[ind] = np.sum(tmp, axis=0)
        self.homophase_est = method.process_outputs(output_coll)

        self.nb_phase = 2*self.N+1
        self.homophase_true = np.zeros((self.nb_phase, self.L),
                                       dtype='complex')
        for p in range(-self.N, self.N+1):
            ind = p % self.nb_phase
            if p:
                start = abs(p)
            else:
                start = 2
            for n in range(start, self.N+1, 2):
                q = (n - p) // 2
                fac = binomial(n, q)
                self.homophase_true[ind] += fac * np.exp(1j * p * phase_vec)

    def test_shape(self):
        self.assertEqual(self.homophase_est.shape, (2*self.N+1, self.L))

    def test_correct_output(self):
        error = rms(self.homophase_est - self.homophase_true)
        self.assertTrue(error < self.tol)


class HPS_NbPhaseTestCase(HPS_Test):

    def setUp(self):
        super().setUp(nb_phase=32)


class HPS_GenInputsTestCase(unittest.TestCase):

    def setUp(self):
        self.N = 3
        self.L = 10000
        self.method = sep.HPS(self.N)

    def test_gen_inputs(self):
        for dtype, out_type in (('float', tuple), ('complex', np.ndarray)):
            with self.subTest(i=dtype):
                input_sig = np.zeros((self.L,), dtype=dtype)
                outputs = self.method.gen_inputs(input_sig)
                self.assertIsInstance(outputs, out_type)


class WarningsNbPhaseTestCase(unittest.TestCase):

    def test_warnings_CPS(self):
        self.assertWarns(UserWarning, sep.CPS, 3, nb_phase=2)

    def test_warnings_HPS(self):
        self.assertWarns(UserWarning, sep.HPS, 3, nb_phase=5)


class ConditionNumberTest(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    method = {'AS': sep.AS,
              'CPS': sep.CPS,
              'PS': sep.PS,
              'PAS': sep.PAS,
              'HPS': sep.HPS}
    L = 10
    test_shape = property()
    test_correct_output = property()

    def setUp(self):
        self.input_dtype['HPS'] = 'complex'
        self.signal_dtype['HPS'] = 'float'
        self.cond_pre = dict()
        self.cond_post = dict()
        for name, method_class in self.method.items():
            input_sig = np.zeros((self.L,), dtype=self.input_dtype[name])
            method = method_class(self.N)
            input_coll = method.gen_inputs(input_sig)
            method.process_outputs(input_coll)
            self.cond_pre[name] = method.condition_numbers
            method.process_outputs(input_coll)
            self.cond_post[name] = method.condition_numbers

    def test_condition_number_cleared(self):
        for name in self.method:
            with self.subTest(i=name):
                self.assertListEqual(self.cond_pre[name], self.cond_post[name])


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
