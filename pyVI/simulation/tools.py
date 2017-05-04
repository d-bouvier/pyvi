# -*- coding: utf-8 -*-
"""
Module for parsing simulation options and parameters for state-sapce systems.

Class
-----
StateSpaceSimulationParameters :
    Regroup all options and parameters required for the simulation of a system
    described with a NumericalStateSpace.


Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 04 May 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

from numpy import identity, dot
from scipy import linalg
from .combinatorics import make_pq_combinatorics


#==============================================================================
# Class
#==============================================================================

class StateSpaceSimulationParameters:
    #TODO docstring

    def __init__(self, A_m, dim_state, mpq: dict, npq: dict, pq_symmetry: bool,
                 **options):
        #TODO docstring

        # Parsing options
        self.fs = options.get('fs', 44100)
        self.nl_order_max = options.get('nl_order_max', 3)
        self.holder_order = options.get('holder_order', 1)
        self.resampling = options.get('resampling', False)

        # Filter matrix
        sampling_time = 1/self.fs
        self.A_inv = linalg.inv(A_m)
        self.filter_mat = linalg.expm(A_m * sampling_time)

        # List of Mpq combinations
        self.mpq_combinatoric = make_pq_combinatorics(mpq, self.nl_order_max,
                                                      pq_symmetry)

        # List of Mpq combinations
        self.npq_combinatoric = make_pq_combinatorics(npq, self.nl_order_max,
                                                      pq_symmetry)

        # Holder bias matrices
        temp_mat = self.filter_mat - identity(dim_state)
        self.holder_bias_mat = dict()
        self.holder_bias_mat[0] = dot(self.A_inv, temp_mat)
        if self.holder_order == 1:
            self.holder_bias_mat[1] = dot(self.A_inv, self.filter_mat) - \
                (1/sampling_time) * dot(dot(self.A_inv, self.A_inv), temp_mat)

