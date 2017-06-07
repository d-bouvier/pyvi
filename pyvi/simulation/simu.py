# -*- coding: utf-8 -*-
"""
Module for numerical simulation of system given its state-space representation.

Function
--------
simulation :
    Compute the simulation of a nonlinear system for a given input.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 07 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from scipy import linalg
from .combinatorics import make_pq_combinatorics
from ..system.statespace import NumericalStateSpace

#==============================================================================
# Class
#==============================================================================

class SimulationObject:
    """
    """
    #TODO docstring

    def __init__(self, system: NumericalStateSpace, fs=44100, nl_order_max=3,
                 holder_order=1, resampling=False):
        """
        """
        #TODO docstring

        # Initialize simulation options
        self.fs = fs
        self.nl_order_max = nl_order_max
        self.holder_order = holder_order
        self.resampling = resampling

        # Filter matrix
        sampling_time = 1/self.fs
        self.A_inv = linalg.inv(system.A_m)
        self.filter_mat = linalg.expm(system.A_m * sampling_time)

        # List of Mpq combinations
        self.mpq_combinatoric = make_pq_combinatorics(system.mpq,
                                                      self.nl_order_max,
                                                      system.pq_symmetry)

        # List of Mpq combinations
        self.npq_combinatoric = make_pq_combinatorics(system.npq,
                                                      self.nl_order_max,
                                                      system.pq_symmetry)

        # Holder bias matrices
        temp_mat_1 = self.filter_mat - np.identity(system.dim['state'])
        temp_mat_2 = np.dot(self.A_inv, temp_mat_1)
        self.holder_bias_mat = dict()
        self.holder_bias_mat[0] = temp_mat_2.copy()
        if self.holder_order == 1:
            self.holder_bias_mat[1] = np.dot(self.A_inv, self.filter_mat) - \
                            (1/sampling_time) * np.dot(self.A_inv, temp_mat_2)

        # Copy system dimensions, matrices and pq-functions
        self.dim = system.dim.copy()

        self.A_m = system.A_m.copy()
        self.B_m = system.B_m.copy()
        self.C_m = system.C_m.copy()
        self.D_m = system.D_m.copy()

        self.mpq = system.mpq.copy()
        self.npq = system.npq.copy()
