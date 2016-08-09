# -*- coding: utf-8 -*-
"""
Description

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Created on Tue Aug  9 16:03:09 2016
Developed for Python 3.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import sympy as sp
from abc import abstractmethod

#==============================================================================
# Functions
#==============================================================================

class StateSpace:
    
    def __init__(self, Am, Bm, Cm, Dm, mpq_dict={}, npq_dict={}, **kwargs):
        # Linear part
        self.Am = Am        
        self.Bm = Bm
        self.Cm = Cm
        self.Dm = Dm        

        # Compute and check system dimensions
        self.dim_state = Am.shape[0]
        self.dim_input = Bm.shape[1]
        self.dim_output = Cm.shape[0]
        self._check_dim_matrices()
        
        # Nonlinear part
        self.mpq = mpq_dict
        self.npq = npq_dict
        self._check_dim_tensor()
        
        self.linear = self._is_linear()
        
    def _check_dim_matrices(self):
        def check_equal(iterator, value):
            return len(set(iterator)) == 1 and iterator[0] == value
             
        list_dim_state = [self.Am.shape[0], self.Am.shape[1],
                          self.Bm.shape[0], self.Cm.shape[1]]     
        list_dim_input = [self.Bm.shape[1], self.Dm.shape[1]]     
        list_dim_output = [self.Cm.shape[0], self.Dm.shape[0]]
        if not check_equal(list_dim_state, self.dim_state):
            raise NameError('State dimension not consistent')
        if not check_equal(list_dim_input, self.dim_input):
            raise NameError('Input dimension not consistent')
        if not check_equal(list_dim_output, self.dim_output):
            raise NameError('Output dimension not consistent')
    
    @abstractmethod
    def _check_dim_tensor(self):
        # not finished
        # must also check good size of inputs and outputs
        for (p, q), mpq in self.mpq.items():
            if mpq.__code__.co_argcount != p + q:
                raise NameError('Wrong number of input arguments for a Mpq ' + \
                                'lambda function\n (got ' + \
                                '{}'.format(mpq.__code__.co_argcount) + \
                                ', expected {})'.format(p + q))
        
    def _is_linear(self):
        return len(self.mpq) == 0 and len(self.npq) == 0
        
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError
    
    # Print 2 LaTeX
    @abstractmethod
    def _print(self):
        raise NotImplementedError        
        
    @abstractmethod
    def simulation(self):
        raise NotImplementedError

    
#==============================================================================
# Functions
#==============================================================================

