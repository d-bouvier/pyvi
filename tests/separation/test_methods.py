# -*- coding: utf-8 -*-
"""
Test script for pyvi/separation/methods.py

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
import pyvi.separation.methods as sep
from pyvi.utilities import binomial, rms


#==============================================================================
# Test Class
#==============================================================================

class _OrderSeparationMethodGlobalTest():

    methods = dict()
    input_dtype = {'AS': 'float',
                   'CPS': 'complex',
                   'RPS': 'complex',
                   'PAS': 'complex'}
    signal_dtype = {'AS': 'float',
                    'CPS': 'complex',
                    'RPS': 'float',
                    'PAS': 'float'}
    true_input_func = {'AS': lambda x: x,
                       'CPS': lambda x: x,
                       'RPS': lambda x: np.real(x),
                       'PAS': lambda x: np.real(x)}
    tol = 5e-10
    N = [4, 5]
    L = 1000

    def setUp(self, **kwargs):
        self.order_est = dict()
        self.order_true = dict()
        self.method_obj = dict()
        self._get_constant_term(**kwargs)
        for method_name, method_class in self.methods.items():
            for N in self.N:
                key = (method_name, N)
                if self.input_dtype[method_name] == 'float':
                    input_sig = np.random.normal(size=(self.L))
                elif self.input_dtype[method_name] == 'complex':
                    input_sig = np.random.normal(size=(self.L)) + \
                                1j * np.random.normal(size=(self.L))
                method_obj = method_class(N, **kwargs)
                input_coll = method_obj.gen_inputs(input_sig)
                output_coll = np.zeros(input_coll.shape,
                                       dtype=self.signal_dtype[method_name])
                for ind in range(input_coll.shape[0]):
                    output_coll[ind] = \
                        generate_output(input_coll[ind], N,
                                        constant_term=self.constant_term)

                self.order_est[key] = method_obj.process_outputs(output_coll)
                self.method_obj[key] = method_obj
                input_sig = self.true_input_func[method_name](input_sig)
                self.order_true[key] = \
                    generate_output(input_sig, N, by_order=True,
                                    constant_term=self.constant_term)

    def _get_constant_term(self, **kwargs):
        self.constant_term = kwargs.get('constant_term', False)

    def test_shape(self):
        for (method, N), val in self.order_est.items():
            with self.subTest(i=(method, N)):
                _N = N+1 if self.constant_term else N
                if isinstance(self.L, tuple):
                    shape = (_N,) + self.L
                else:
                    shape = (_N, self.L)
                self.assertEqual(val.shape, shape)

    def test_correct_output(self):
        for (method, N), val in self.order_est.items():
            with self.subTest(i=(method, N)):
                error = rms(val - self.order_true[(method, N)])
                self.assertTrue(error < self.tol)


class NoKwargsTestCase(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    methods = {'AS': sep.AS,
               'CPS': sep.CPS,
               'RPS': sep.RPS,
               'PAS': sep.PAS}


class AmplitudesTestCase(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    methods = {'AS': sep.AS,
               'PAS': sep.PAS}
    amplitudes = [1, 0.8, 0.5, 0.2, 0.1]

    def setUp(self):
        super().setUp(amplitudes=self.amplitudes)

    def test_correct_amplitudes(self):
        for (name, N), method in self.method_obj.items():
            with self.subTest(i=(name, N)):
                self.assertListEqual(method.amplitudes, self.amplitudes)


class GainAS_TestCase(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    methods = {'AS': sep.AS}
    gain = 2.
    amplitudes = {4: [1., -1., 2., -2.],
                  5: [1., -1., 2., -2., 4.]}
    tol = 1e-10

    def setUp(self):
        super().setUp(gain=self.gain)

    def test_correct_amplitudes(self):
        for (name, N), method in self.method_obj.items():
            with self.subTest(i=(name, N)):
                correct = np.all(method.amplitudes == self.amplitudes[N])
                self.assertTrue(correct)


class GainPAS_TestCase(GainAS_TestCase):

    methods = {'PAS': sep.PAS}
    gain = 2.
    amplitudes = {4: [1., 2.],
                  5: [1., 2., 4.]}


class NegativeAmpTestCase(GainAS_TestCase):

    methods = {'AS': sep.AS}
    amplitudes = {4: [1., np.cos(np.pi/7), np.cos(2*np.pi/7),
                      np.cos(3*np.pi/7)],
                  5: [1., np.cos(np.pi/9), np.cos(2*np.pi/9),
                      np.cos(3*np.pi/9), np.cos(4*np.pi/9)]}

    def setUp(self):
        _OrderSeparationMethodGlobalTest.setUp(self, negative_amp=False)



class NegativeAmpAndGainTestCase(GainAS_TestCase):

    methods = {'AS': sep.AS}
    gain = 2.
    amplitudes = {4: [1., 2., 4., 8.],
                  5: [1., 2., 4., 8., 16.]}
    tol = 1e-6

    def setUp(self):
        _OrderSeparationMethodGlobalTest.setUp(self, negative_amp=False,
                                               gain=self.gain)


class NbAmpTestCase(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    methods = {'AS': sep.AS,
               'PAS': sep.PAS}

    def setUp(self):
        super().setUp(nb_amp=3*max(self.N))


class NbPhaseTestCase(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    methods = {'CPS': sep.CPS,
               'RPS': sep.RPS,
               'PAS': sep.PAS}
    tol = 5e-10

    def setUp(self):
        super().setUp(nb_phase=32)


class RhoTestCase(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    methods = {'CPS': sep.CPS}
    atol = {'CPS': 5e-10}

    def setUp(self):
        super().setUp(rho=2.)


class ConstantTermTestCase(_OrderSeparationMethodGlobalTest,
                           unittest.TestCase):

    methods = {'AS': sep.AS,
               'CPS': sep.CPS,
               'RPS': sep.RPS,
               'PAS': sep.PAS}

    def setUp(self):
        super().setUp(constant_term=True)


class HPS_Test(_OrderSeparationMethodGlobalTest, unittest.TestCase):

    method = sep.HPS
    tol = 1e-14
    N = 5
    nb_phase = 2*N+1
    shape = (nb_phase, _OrderSeparationMethodGlobalTest.L)

    def _create_phase_vec(self):
        return 2 * np.pi * np.arange(self.L)/self.L

    def _init_homophase_true(self):
        self.homophase_true = np.zeros((self.nb_phase, self.L),
                                       dtype='complex')

    def setUp(self, **kwargs):
        phase_vec = self._create_phase_vec()
        input_sig = np.exp(1j * phase_vec)
        power_vec = np.arange(1, self.N+1)

        method_obj = self.method(self.N, **kwargs)
        input_coll = method_obj.gen_inputs(input_sig)
        output_coll = np.zeros(input_coll.shape, dtype='complex')

        for ind in range(input_coll.shape[0]):
            slice_obj = (slice(None),) + (np.newaxis,)*(phase_vec.ndim)
            tmp = input_coll[ind][np.newaxis, :]**power_vec[slice_obj]
            output_coll[ind] = np.sum(tmp, axis=0)
        self.homophase_est = method_obj.process_outputs(output_coll)

        self._init_homophase_true()
        for p in range(-self.N, self.N+1):
            ind = p % self.nb_phase
            if p:
                start = abs(p)
            else:
                start = 2
            for n in range(start, self.N+1, 2):
                q = (n - p) // 2
                fac = binomial(n, q) / 2**n
                self.homophase_true[ind] += fac * np.exp(1j * p * phase_vec)

    def test_shape(self):
        self.assertEqual(self.homophase_est.shape, self.shape)

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
        for dtype, return_cplx, out_type in (('float', True, tuple),
                                             ('float', False, np.ndarray),
                                             ('complex', True, np.ndarray),
                                             ('complex', False, np.ndarray)):
            with self.subTest(i=(dtype, return_cplx)):
                input_sig = np.zeros((self.L), dtype=dtype)
                outputs = self.method.gen_inputs(input_sig,
                                                 return_cplx_sig=return_cplx)
                self.assertIsInstance(outputs, out_type)


class RPS_RawModeTestCase(unittest.TestCase):

    method = sep.RPS
    N = 3
    L = 1000

    def setUp(self):
        method = self.method(self.N)
        output_coll = np.zeros((method.K, self.L))
        _, self.term_est = method.process_outputs(output_coll, raw_mode=True)

    def test_keys(self):
        keys = dict()
        for n in range(1, self.N+1):
            for k in range(n//2 + 1):
                keys[(n, k)] = ()
        self.assertEqual(self.term_est.keys(), keys.keys())


class PAS_RawModeTestCase(RPS_RawModeTestCase):

    method = sep.PAS


class WarningsNbAmpTestCase(unittest.TestCase):

    def test_warnings_AS(self):
        self.assertWarns(UserWarning, sep.AS, 3, nb_amp=2)


class WarningsNbPhaseTestCase(unittest.TestCase):

    def test_warnings_CPS(self):
        self.assertWarns(UserWarning, sep.CPS, 3, nb_phase=2)

    def test_warnings_HPS(self):
        self.assertWarns(UserWarning, sep.HPS, 3, nb_phase=5)


class ErrorASNotEnoughFactorsTestCase(unittest.TestCase):

    method = sep._SeparationMethod(3, [])

    def test_error_raised(self):
        self.assertRaises(ValueError, sep.AS, 3, amplitudes=[1, 0.5])


class ErrorMultiDimInputTestCase(unittest.TestCase):

    method = sep._SeparationMethod(3, [])

    def test_error_raised(self):
        for shape in [(100, 2), (2, 100), (1, 100, 2)]:
            self.assertRaises(ValueError, self.method.gen_inputs,
                              np.ones(shape))

    def test_error_not_raised(self):
        for shape in [(100,), (1, 100), (100, 1), (1, 100, 1)]:
            try:
                self.method.gen_inputs(np.ones(shape))
            except ValueError:
                self.fail()


class _ConditionNumbersGlobalTest():

    method = sep._SeparationMethod
    N = 3
    p_values = [None, 1, -1, 2, -2, np.inf, -np.inf, 'fro']
    results = {}
    kwargs = {}
    tol = 1e-8

    def setUp(self):
        self.obj = self.method(self.N, **self.kwargs)

    def test_correct(self):
        for p in self.p_values:
            with self.subTest(i=p):
                val = np.array(self.obj.get_condition_numbers(p=p))
                ref = np.array(self.results[p])
                self.assertTrue(np.allclose(val, ref, atol=self.tol, rtol=0))


class AS_ConditionNumbersTest(_ConditionNumbersGlobalTest, unittest.TestCase):

    method = sep.AS
    results = {None: [8.04346051],
               1: [13. + 1/3],
               -1: [2.125],
               2: [8.04346051],
               -2: [0.12432459],
               np.inf: [12],
               -np.inf: [0.875],
               'fro': [10.09718401]}
    kwargs = {'gain': 0.5}


class CPS_ConditionNumbersTest(_ConditionNumbersGlobalTest, unittest.TestCase):

    method = sep.CPS
    nb_phase = 10
    results = {None: [1.],
               1: [nb_phase],
               -1: [nb_phase],
               2: [1.],
               -2: [1.],
               np.inf: [nb_phase],
               -np.inf: [nb_phase],
               'fro': [1.]}
    kwargs = {'nb_phase': nb_phase}


class CPS_ConditionNumbersTestWithRho(CPS_ConditionNumbersTest,
                                      unittest.TestCase):

    nb_phase = 10
    results = {None: [1., 100.],
               1: [nb_phase, 100.],
               -1: [nb_phase, 0.01],
               2: [1., 100.],
               -2: [1., 0.01],
               np.inf: [nb_phase, 100.],
               -np.inf: [nb_phase, 0.01],
               'fro': [1., 101.01]}
    kwargs = {'nb_phase': nb_phase,
              'rho': 0.1}


class HPS_ConditionNumbersTest(_ConditionNumbersGlobalTest, unittest.TestCase):

    method = sep.HPS
    nb_phase = 11
    results = {None: [1.],
               1: [nb_phase],
               -1: [nb_phase],
               2: [1.],
               -2: [1.],
               np.inf: [nb_phase],
               -np.inf: [nb_phase],
               'fro': [1.]}
    kwargs = {'nb_phase': nb_phase}


class RPS_ConditionNumbersTest(_ConditionNumbersGlobalTest, unittest.TestCase):

    method = sep.RPS
    nb_phase = 11
    results = {None: [1., 1., 8.71863649, 1., 1.],
               1: [nb_phase**2, 1+1/3, 13, 1+1/3, 1.2],
               -1: [nb_phase**2, 2/3, 1, 2/3, .4],
               2: [1., 1., 8.718636499713, 1., 1.],
               -2: [1., 1., 0.114696833619, 1., 1.],
               np.inf: [nb_phase**2, 1+1/3, 13., 1+1/3, 1.2],
               -np.inf: [nb_phase**2, 2/3, 1., 2/3, .4],
               'fro': [1., 1., 17+2/3, 2., 2.]}
    kwargs = {'nb_phase': nb_phase}


class PAS_ConditionNumbersTest(_ConditionNumbersGlobalTest, unittest.TestCase):

    method = sep.PAS
    nb_phase = 11
    results = {None: [1., 1., 9.12654057, 1., 1.],
               1: [nb_phase, 1.17647059, 12, 1.17647059, 1.10769231],
               -1: [nb_phase, 0.29411765, 1.16666667, 0.29411765, 0.13846154],
               2: [1., 1., 9.12654057, 1., 1.],
               -2: [1., 1., 0.10957054, 1., 1.],
               np.inf: [nb_phase, 1.17647059, 12, 1.17647059,
                        1.10769231],
               -np.inf: [nb_phase, 0.29411765, 1.16666667, 0.29411765,
                         0.13846154],
               'fro': [1., 1., 18.47222222, 2., 2.]}
    kwargs = {'nb_phase': nb_phase}
    tol = 1e-8


class AS_OptimumGainTest(unittest.TestCase):

    best_gains = [(3, {}, 0.526635),
                  (3, {'negative_amp': True}, 0.526629),
                  (3, {'negative_amp': False}, 0.539772),
                  (3, {'nb_amp': 10}, 0.664619),
                  (9, {}, 0.791742),
                  (3, {'constant_term': True}, 0.521798),
                  (3, {'negative_amp': True, 'constant_term': True},
                   0.521798),
                  (3, {'negative_amp': False, 'constant_term': True},
                   0.473819),
                  (3, {'nb_amp': 10, 'constant_term': True}, 0.726372),
                  (9, {'constant_term': True}, 0.794541)]
    tol = 1e-4
    tol_method = 1e-6
    method = sep.AS

    def test_correct(self):
        for N, kwargs, ref in self.best_gains:
            with self.subTest(i=(N, kwargs)):
                val = self.method.optimum_gain(N, tol=self.tol_method,
                                               **kwargs)
                error = abs(ref - val)
                self.assertTrue(error < self.tol)


class PAS_OptimumGainTest(AS_OptimumGainTest):

    best_gains = [(3, {}, 0.538955),
                  (4, {}, 0.672761),
                  (5, {}, 0.646210),
                  (3, {'constant_term': True}, 0.538954),
                  (4, {'constant_term': True}, 0.539649),
                  (5, {'constant_term': True}, 0.646210)]
    method = sep.PAS


#==============================================================================
# Functions
#==============================================================================

def generate_output(input_sig, N, by_order=False, constant_term=False):
    output_by_order = np.zeros((N,) + input_sig.shape, dtype=input_sig.dtype)
    for n in range(N):
        output_by_order[n, :] = input_sig**(n+1)
        output_by_order[n, 1:] += input_sig[:-1]**(n+1)
        output_by_order[n, 1:] -= 2*input_sig[:-1]**n * input_sig[1:]
    if constant_term:
        _term_cst = np.ones((1, len(input_sig)))
        output_by_order = np.concatenate((_term_cst, output_by_order), axis=0)
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
