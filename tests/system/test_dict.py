# -*- coding: utf-8 -*-
"""
Test script for pyvi/system/dict.py

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

#==============================================================================
# Importations
#==============================================================================

import unittest
import pyvi.system.dict as systems


#==============================================================================
# Test Class
#==============================================================================

class CreateLoudspeakerTestCase(unittest.TestCase):

    def setUp(self):
        self.system = systems.create_loudspeaker_sica()

    def test_dim_input(self):
        self.assertEqual(self.system.dim['input'], 1)

    def test_dim_state(self):
        self.assertEqual(self.system.dim['state'], 3)

    def test_dim_output(self):
        self.assertEqual(self.system.dim['output'], 1)

    def test_siso(self):
        self.assertEqual(self.system._type, 'SISO')

    def test_not_linear(self):
        self.assertFalse(self.system.linear)

    def test_output_eqn_linear(self):
        self.assertTrue(self.system._out_eqn_linear)

    def test_dyn_nl_only_on_state(self):
        self.assertTrue(self.system._dyn_nl_only_on_state)

    def test_not_nl_colinear(self):
        self.assertFalse(self.system._nl_colinear)


class CreateNlDampingTestCase(unittest.TestCase):

    def setUp(self):
        self.system = systems.create_nl_damping()

    def test_dim_input(self):
        self.assertEqual(self.system.dim['input'], 1)

    def test_dim_state(self):
        self.assertEqual(self.system.dim['state'], 2)

    def test_dim_output(self):
        self.assertEqual(self.system.dim['output'], 1)

    def test_attribute_siso(self):
        self.assertEqual(self.system._type, 'SISO')

    def test_not_linear(self):
        self.assertFalse(self.system.linear)

    def test_out_eqn_linear(self):
        self.assertTrue(self.system._out_eqn_linear)

    def test_dyn_nl_only_on_state(self):
        self.assertTrue(self.system._dyn_nl_only_on_state)

    def test_nl_colinear(self):
        self.assertTrue(self.system._nl_colinear)


class CreateMoogTestCase(unittest.TestCase):

    def setUp(self):
        self.system = systems.create_moog(taylor_series_truncation=3)
        self.mpq = {(3, 0): 0, (2, 1): 0, (1, 2): 0, (0, 3): 0}

    def test_dim_input(self):
        self.assertEqual(self.system.dim['input'], 1)

    def test_dim_state(self):
        self.assertEqual(self.system.dim['state'], 4)

    def test_dim_output(self):
        self.assertEqual(self.system.dim['output'], 1)

    def test_siso(self):
        self.assertEqual(self.system._type, 'SISO')

    def test_not_linear(self):
        self.assertFalse(self.system.linear)

    def test_out_eqn_linear(self):
        self.assertTrue(self.system._out_eqn_linear)

    def test_not_nl_colinear(self):
        self.assertFalse(self.system._nl_colinear)

    def test_right_mpqs(self):
        self.assertEqual(self.system.mpq.keys(), self.mpq.keys())


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    unittest.main()

