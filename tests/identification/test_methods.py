# -*- coding: utf-8 -*-
"""
Test script for pyvi/identification/methods.py

Notes
-----
Developed for Python 3.6
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
from pyvi.separation.methods import HPS, RPS
from pyvi.volterra.combinatorial_basis import (_check_parameters,
                                               _compute_list_nb_coeff,
                                               compute_combinatorial_basis)
from pyvi.utilities.orthogonal_basis import LaguerreBasis


#==============================================================================
# Test Class
#==============================================================================

class DirectMethodTest(unittest.TestCase):

    N = 4
    L = 100
    rtol = 0
    atol = 1e-12
    method = staticmethod(direct_method)
    solvers = {'LS', 'QR'}
    cast_modes = {'real', 'imag', 'real-imag'}
    sigma = 1.

    def _set_kwargs(self):
        return {'M': 3}

    def _generate_kernels(self):
        return generate_kernels(self.N, **self.kwargs)

    def _create_input(self):
        return np.random.normal(scale=self.sigma, size=(self.L,))

    def _create_output(self, input_sig):
        return generate_output(input_sig, self.kernels_vec, self.N,
                               **self.kwargs)

    def _identification(self):
        list_kernels_est = dict()
        for solver, cast_mode in itr.product(self.solvers, self.cast_modes):
            list_kernels_est[(solver, cast_mode)] = \
                self.method(self.input_sig, self.output_data, self.N,
                            solver=solver, cast_mode=cast_mode, **self.kwargs)
        return list_kernels_est

    def setUp(self):
        self.kwargs = self._set_kwargs()
        self.kernels_vec, self.length = self._generate_kernels()
        self.input_sig = self._create_input()
        self.output_data = self._create_output(self.input_sig)
        self.list_kernels_est = self._identification()

    def test_check_keys_dict(self):
        keys = set(range(1, self.N+1))
        for key, kernels_est in self.list_kernels_est.items():
            with self.subTest(i=key):
                self.assertSetEqual(set(kernels_est.keys()), keys)

    def test_check_shape_kernels(self):
        for key, kernels_est in self.list_kernels_est.items():
            for n, h in kernels_est.items():
                with self.subTest(i=(n, key)):
                    self.assertEqual(h.shape, (self.length[n-1],))

    def test_check_kernels_are_real(self):
        for key, kernels_est in self.list_kernels_est.items():
            for n, h in kernels_est.items():
                with self.subTest(i=(n, key)):
                    if not np.isrealobj(h):
                        print(h.dtype, np.isrealobj(h), (n, key))
                    self.assertTrue(np.isrealobj(h))

    def test_correct_output(self):
        for key, kernels_est in self.list_kernels_est.items():
            for n, h in kernels_est.items():
                with self.subTest(i=(n, key)):
                    self.assertTrue(np.allclose(h, self.kernels_vec[n],
                                                rtol=self.rtol,
                                                atol=self.atol))


class OrderMethodTest(DirectMethodTest):

    method = staticmethod(order_method)

    def _create_output(self, input_sig):
        return generate_output(input_sig, self.kernels_vec, self.N,
                               by_order=True, **self.kwargs)


class TermMethodTest(DirectMethodTest):

    method = staticmethod(term_method)

    def _create_input(self):
        return super()._create_input() + 1j * super()._create_input()

    def _create_output(self, input_sig):
        sep_method = RPS(self.N)
        input_coll = sep_method.gen_inputs(input_sig)
        output_coll = np.zeros(input_coll.shape)
        for ind in range(input_coll.shape[0]):
            output_coll[ind] = super()._create_output(input_coll[ind])
        _, output_data = sep_method.process_outputs(output_coll, raw_mode=True)
        return output_data


class IterMethodTest(TermMethodTest):

    method = staticmethod(iter_method)

    def _create_output(self, input_sig):
        sep_method = HPS(self.N)
        input_coll = sep_method.gen_inputs(input_sig)
        output_coll = np.zeros(input_coll.shape)
        for ind in range(input_coll.shape[0]):
            output_coll[ind] = generate_output(input_coll[ind],
                                               self.kernels_vec, self.N,
                                               **self.kwargs)
        return sep_method.process_outputs(output_coll)


class PhaseMethodTest(IterMethodTest):

    method = staticmethod(phase_method)


class DirectMethod_ListM_Test(DirectMethodTest):

    def _set_kwargs(self):
        return {'M': [3, 5, 0, 5]}


class OrderMethod_ListM_Test(OrderMethodTest, DirectMethod_ListM_Test):
    pass


class TermMethod_ListM_Test(TermMethodTest, DirectMethod_ListM_Test):
    pass


class IterMethod_ListM_Test(IterMethodTest, DirectMethod_ListM_Test):
    pass


class PhaseMethod_ListM_Test(PhaseMethodTest, DirectMethod_ListM_Test):
    pass


class DirectMethod_Projected_Test(DirectMethodTest):

    def _set_kwargs(self):
        return {'orthogonal_basis': LaguerreBasis(0.01, 3)}


class OrderMethod_Projected_Test(OrderMethodTest, DirectMethod_Projected_Test):
    pass


class TermMethod_Projected_Test(TermMethodTest, DirectMethod_Projected_Test):
    pass


class IterMethod_Projected_Test(IterMethodTest, DirectMethod_Projected_Test):
    pass


class PhaseMethod_Projected_Test(PhaseMethodTest, DirectMethod_Projected_Test):
    pass


class DirectMethod_MultiProj_Test(DirectMethodTest):

    def _set_kwargs(self):
        return {'orthogonal_basis': [LaguerreBasis(0.01, 3),
                                     LaguerreBasis(0.01, 5),
                                     LaguerreBasis(0.01, 0),
                                     LaguerreBasis(0.01, 5)]}


class OrderMethod_MultiProj_Test(OrderMethodTest, DirectMethod_MultiProj_Test):
    pass


class TermMethod_MultiProj_Test(TermMethodTest, DirectMethod_MultiProj_Test):
    pass


class IterMethod_MultiProj_Test(IterMethodTest, DirectMethod_MultiProj_Test):
    pass


class PhaseMethod_MultiProj_Test(PhaseMethodTest, DirectMethod_MultiProj_Test):
    pass


class DirectMethodHammersteinTest(DirectMethodTest):

    def _set_kwargs(self):
        return {'M': 3, 'system_type': 'hammerstein'}


class OrderMethodHammersteinTest(DirectMethodHammersteinTest, OrderMethodTest):
    pass


class TermMethodHammersteinTest(DirectMethodHammersteinTest, TermMethodTest):
    pass


class IterMethodHammersteinTest(DirectMethodHammersteinTest, IterMethodTest):
    pass


class PhaseMethodHammersteinTest(DirectMethodHammersteinTest, PhaseMethodTest):
    pass


class DirectMethodHammerstein_ListM_Test(DirectMethodHammersteinTest,
                                         DirectMethod_ListM_Test):
    pass


class OrderMethodHammerstein_ListM_Test(DirectMethodHammerstein_ListM_Test,
                                        OrderMethodTest):
    pass


class TermMethodHammerstein_ListM_Test(DirectMethodHammerstein_ListM_Test,
                                       TermMethodTest):
    pass


class IterMethodHammerstein_ListM_Test(DirectMethodHammerstein_ListM_Test,
                                       IterMethodTest):
    pass


class PhaseMethodHammerstein_ListM_Test(DirectMethodHammerstein_ListM_Test,
                                        PhaseMethodTest):
    pass



class DirectMethodHammerstein_Proj_Test(DirectMethodHammersteinTest,
                                        DirectMethod_Projected_Test):
    pass


class OrderMethodHammerstein_Proj_Test(DirectMethodHammerstein_Proj_Test,
                                       OrderMethodTest):
    pass


class TermMethodHammerstein_Proj_Test(DirectMethodHammerstein_Proj_Test,
                                      TermMethodTest):
    pass


class IterMethodHammerstein_Proj_Test(DirectMethodHammerstein_Proj_Test,
                                      IterMethodTest):
    pass


class PhaseMethodHammerstein_Proj_Test(DirectMethodHammerstein_Proj_Test,
                                       PhaseMethodTest):
    pass


class HammersteinWarningTest(unittest.TestCase):

    def test_warning(self):
        self.assertWarns(UserWarning, direct_method, np.arange(30),
                         np.arange(30), 3, M=5, out_form='tri',
                         system_type='hammerstein')


#==============================================================================
# Functions
#==============================================================================

def generate_output(input_sig, kernels_vec, N, M=None, orthogonal_basis=None,
                    system_type='volterra', by_order=False):
    phi = compute_combinatorial_basis(input_sig, N, system_type=system_type,
                                      M=M, orthogonal_basis=orthogonal_basis,
                                      sorted_by='order')
    L = phi[1].shape[0]
    output_by_order = np.zeros((N, L))
    for n in range(N):
        output_by_order[n, :] = np.dot(phi[n+1], kernels_vec[n+1])
    if by_order:
        return output_by_order
    else:
        return np.sum(output_by_order, axis=0)


def generate_kernels(N, M=None, orthogonal_basis=None, system_type='volterra'):
    _M, is_orthogonal_basis_as_list = _check_parameters(N, system_type, M,
                                                        orthogonal_basis)
    list_nb_coeff = _compute_list_nb_coeff(N, system_type, _M,
                                           orthogonal_basis,
                                           is_orthogonal_basis_as_list)
    kernels_vec = dict()
    for indn, nb_coeff in enumerate(list_nb_coeff):
        kernels_vec[indn+1] = np.random.uniform(low=-1., high=1.,
                                                size=nb_coeff)
    return kernels_vec, list_nb_coeff


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
