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


#==============================================================================
# Test Class
#==============================================================================

class _SeparationMethodGlobalTest():

    methods = dict()
    input_dtype = {'AS': 'float',
                   'CPS': 'complex',
                   'HPS': 'complex',
                   'PAS': 'complex'}
    signal_dtype = {'AS': 'float',
                    'CPS': 'complex',
                    'HPS': 'float',
                    'PAS': 'float'}
    true_input_func = {'AS': lambda x: x,
                       'CPS': lambda x: x,
                       'HPS': lambda x: 2 * np.real(x),
                       'PAS': lambda x: 2 * np.real(x)}
    atol = {'AS': 1e-11,
            'CPS': 5e-12,
            'HPS': 1e-12,
            'PAS': 1e-10}
    N = {'AS': 5,
         'CPS': 5,
         'HPS': 2,
         'PAS': 5}
    L = 10000

    def setUp(self, **kwargs):
        self.order_est = dict()
        self.order_true = dict()
        for name, method_class in self.methods.items():
            if self.input_dtype[name] == 'float':
                input_sig = np.random.normal(size=(self.L,))
            elif self.input_dtype[name] == 'complex':
                input_sig = np.random.normal(size=(self.L,)) + \
                            1j * np.random.normal(size=(self.L,))
            method = method_class(self.N[name], **kwargs)
            input_coll = method.gen_inputs(input_sig)
            output_coll = np.zeros(input_coll.shape,
                                   dtype=self.signal_dtype[name])
            for ind in range(input_coll.shape[0]):
                output_coll[ind] = generate_output(input_coll[ind],
                                                   self.N[name])
            if name == 'HPS':
                temp = method.process_outputs(output_coll)
                self.order_est[name] = np.stack((temp[1]+temp[-1],
                                                 temp[0]+temp[2]+temp[-2]),
                                                axis=0)
            else:
                self.order_est[name] = method.process_outputs(output_coll)
            input_sig = self.true_input_func[name](input_sig)
            self.order_true[name] = generate_output(input_sig, self.N[name],
                                                    by_order=True)

    def test_shape(self):
        for name in self.methods:
            with self.subTest(i=name):
                self.assertEqual(self.order_est[name].shape,
                                 (self.N[name], self.L))

    def test_correct_output(self):
        for name in self.methods:
            with self.subTest(i=name):
                self.assertTrue(np.allclose(self.order_est[name],
                                            self.order_true[name],
                                            rtol=0, atol=self.atol[name]))


class NoKwargsTestCase(_SeparationMethodGlobalTest, unittest.TestCase):

    methods = {'AS': sep.AS,
               'CPS': sep.CPS,
               'HPS': sep.HPS,
               'PAS': sep.PAS}


class GainTestCase(_SeparationMethodGlobalTest, unittest.TestCase):

    methods = {'AS': sep.AS,
               'PAS': sep.PAS}
    atol = {'AS': 1e-10,
            'PAS': 1e-8}

    def setUp(self):
        super().setUp(gain=1.5)


class NegativeGainTestCase(_SeparationMethodGlobalTest, unittest.TestCase):

    methods = {'AS': sep.AS}
    atol = {'AS': 5e-10}

    def setUp(self):
        super().setUp(negative_gain=False)


class K_ArgumentTestCase(_SeparationMethodGlobalTest, unittest.TestCase):

    methods = {'AS': sep.AS}

    def setUp(self):
        super().setUp(K=10)


class NbPhaseTestCase(_SeparationMethodGlobalTest, unittest.TestCase):

    methods = {'CPS': sep.CPS,
               'HPS': sep.HPS,
               'PAS': sep.PAS}

    def setUp(self):
        super().setUp(nb_phase=32)



class RhoTestCase(_SeparationMethodGlobalTest, unittest.TestCase):

    methods = {'CPS': sep.CPS,
               'HPS': sep.HPS,
               'PAS': sep.PAS}
    atol = {'CPS': 5e-10,
            'HPS': 1e-12,
            'PAS': 5e-10}

    def setUp(self):
        super().setUp(rho=2.)


class WarningsNbPhaseTestCase(unittest.TestCase):

    def test_warnings_CPS(self):
        self.assertWarns(UserWarning, sep.CPS, 3, nb_phase=2)

    def test_warnings_HPS(self):
        self.assertWarns(UserWarning, sep.HPS, 3, nb_phase=5)



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


class PAS_RawModeTestCase(unittest.TestCase):

    def setUp(self):
        self.N = 3
        self.L = 10000
        method = sep.PAS(self.N)
        output_coll = np.zeros((method.K, self.L))
        _, self.term_est = method.process_outputs(output_coll, raw_mode=True)

    def test_keys(self):
        keys = dict()
        for n in range(1, self.N+1):
            for k in range(n//2 + 1):
                keys[(n, k)] = ()
        self.assertEqual(self.term_est.keys(), keys.keys())


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
