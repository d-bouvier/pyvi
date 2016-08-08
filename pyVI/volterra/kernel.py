# -*- coding: utf-8 -*-
"""
Description

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Created on Mon Aug  8 10:22:17 2016
Developed for Python 3.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import sympy as sp
import numpy as np
from abc import abstractmethod

#==============================================================================
# Class
#==============================================================================

class Kernel:
    """ Class that defines a Volterra kernel described by:
    - its expression
    - its order of nonlinearity
    Also defines if it's a symmetric kernel and the list of sympy symbols used 
    in the expression."""
    
    def __init__(self, expr=sp.Integer(1), order=1, **kwargs):
        self.expr = expr
        self.order = order

        if 'symmetric' in kwargs:
            self.symmetric = kwargs['symmetric']
        if 'symbols' in kwargs:
            self.symbols = kwargs['symbols']
        else:
            self.symbols = sp.var('H{} s(1:{})'.format(self.order,
                                                       self.order + 1))
      
    
    def __repr__(self):
        repr_str = sp.pretty(self.symbols[0])        
        repr_str += sp.pretty(self.symbols[1:])
        repr_str += ' = '
        repr_str += sp.pretty(self.expr)
        return repr_str
        
    def __print__(self):
        repr_str = 'Volterra kernel of order {}:'.format(self.order)
        repr_str += '\n'
        repr_str += self.__repr__()
        return repr_str

    @abstractmethod
    def plot(self):
        return NotImplementedError

    @abstractmethod
    def symmetrize(self):
        return NotImplementedError

    @abstractmethod
    def regularize(self):
        return NotImplementedError

    @abstractmethod
    def triangularize(self):
        return NotImplementedError

#==============================================================================
# Functions
#==============================================================================

