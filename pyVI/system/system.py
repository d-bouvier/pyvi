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
    
    
    def __init__(self, Am, Bm, Cm, Dm, mpq_dict={}, npq_dict={}, **kwargs):
        """Initialize the system."""

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

