# -*- coding: utf-8 -*-
"""
Test script for pyvi/simulation/simu.py

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 24 Nov. 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import unittest
import itertools
import numpy as np
from pyvi.system.dict import create_test
from pyvi.simulation.simu import SimulationObject


#==============================================================================
# Test Class
#==============================================================================

class SimulationTestCase(unittest.TestCase):

    def setUp(self):
        self.len = 2000
        self.fs = 100
        self.sig = np.ones((self.len,))
        self.system = create_test(mode='numeric')
        self.n_state = self.system.dim['state']
        self.n_out = self.system.dim['output']
        self.list_order = [1, 2, 3, 4]
        self.list_out_opt = ['output', 'output_by_order', 'state',
                             'state_by_order']
        self.list_holder_order = [0, 1]
        self.list_resampling = [True, False]

    def test_output_shape(self):
        loop = itertools.product(self.list_order, self.list_out_opt,
                                 self.list_holder_order, self.list_resampling)
        for i, (N, out_opt, holder_order, resampling) in enumerate(loop):
            with self.subTest(i=i):
                simuObj = SimulationObject(self.system, fs=self.fs,
                                           nl_order_max=N,
                                           holder_order=holder_order,
                                           resampling=resampling)
                out = simuObj.simulation(self.sig, out_opt=out_opt)
                correct_out_shape = (self.len,)
                if ('output' in out_opt) & (self.n_out != 1):
                    correct_out_shape = (self.n_out,) + correct_out_shape
                elif 'state' in out_opt:
                    correct_out_shape = (self.n_state,) + correct_out_shape
                if 'by_order' in out_opt:
                    correct_out_shape = (N,) + correct_out_shape
                self.assertTupleEqual(out.shape, correct_out_shape)


class ComputeKernelsTestCase(unittest.TestCase):

    def setUp(self):
        options = {'fs': 2000,
                   'nl_order_max': 2,
                   'holder_order': 1}
        self.sys_simu = SimulationObject(create_test(mode='numeric'),
                                         **options)
        self.T = 0.03
        self.which_options = ['time', 'freq', 'both']

    def test_compute_kernels(self):
        for opt in self.which_options:
            with self.subTest(i=opt):
                output = self.sys_simu.compute_kernels(self.T, which=opt)
            self.assertIsNotNone(output)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()
