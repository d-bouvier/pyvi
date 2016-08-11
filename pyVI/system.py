# -*- coding: utf-8 -*-
"""
Description

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Created on Wed Aug 10 16:31:01 2016
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

class System:
    """Object-oriented symbolic representation of a system
    
    This class represents a system (linear or nonlinear) using either its
    state-space representation (mathematical representation used in control
    engineering, see https://en.wikipedia.org/wiki/State-space_representation)
    or its Volterra series (see https://en.wikipedia.org/wiki/Volterra_series).
    
    Relies on the Sympy module.
    
    Attributes
    ----------

    """
    
    
    def __init__(self, label='Test', *args, **kwargs):
        """Initialize the system."""
        self.label = label
        self.dim_input = kwargs.get('dim_input', None)
        self.dim_output = kwargs.get('dim_output', None)
        
        self._has_state_space_repr = False
        self._has_volterra = False
        
        from symbols.symbols import Symbols
        self.symbols = Symbols(3)
    
    
    def add_state_space_repr(self, Am, Bm, Cm, Dm, mpq={}, npq={}):
        from system.statespace import StateSpace
        self.state_space = StateSpace(Am, Bm, Cm, Dm, mpq, npq)

        self.dim_input = self.state_space.dim_input
        self.dim_output = self.state_space.dim_output
        self._has_state_space_repr = True
        
    
    def __repr__(self):
        repr_str = ''
        if self._has_state_space_repr:
            repr_str += 'State-space representation\n'
            repr_str += self.state_space.__repr__()
        if self._has_volterra:
            repr_str += 'Volterra series\n'
            repr_str += self.volterra.__repr__()
        return repr_str
    
    
    @abstractmethod    
    def _is_linear(self):
        """Check if the system is linear."""
        raise NotImplementedError

    @abstractmethod
    def _is_passive(self):
        """Check if the system is passive."""
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

