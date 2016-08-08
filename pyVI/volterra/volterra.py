# -*- coding: utf-8 -*-
"""
Description

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Created on Mon Aug  8 12:10:11 2016
Developed for Python 3.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import sympy as sp
import numpy as np
from kernel import Kernel
from abc import abstractmethod

#==============================================================================
# Class
#==============================================================================

class Volterra:
    """ Class that defines Volterra serie of a system, described by:
    - its expression
    - its order of nonlinearity
    Also defines if it's a symmetric kernel and the list of sympy symbols used 
    in the expression."""
    
    def __init__(self, kernels=[Kernel()], **kwargs):
        self.kernels = kernels
        self.list_kernels = []
        for idx, kernel in enumerate(self.kernels):
            self.list_kernels.append(kernel.order)
                
        if 'order_max' in kwargs:
            self.order_max = kwargs['order_max']
        else:
            self.order_max = max(self.list_kernels)    
        if 'symmetric' in kwargs:
            self.symmetric = kwargs['symmetric']
        else:
            self.symmetric = True
            for idx, kernel in enumerate(self.kernels):
                self.symmetric = self.symmetric and kernel.symmetric
        if 'symbols' in kwargs:
            self.symbols = kwargs['symbols']
        else:
            self.symbols = sp.var('H s(1:{})'.format(self.order_max + 1))
      
    
    def __repr__(self):
        repr_str = ''
        for idx, kernel in enumerate(self.kernels):
            repr_str += kernel.__repr__()
            repr_str += '\n'
        return repr_str
    
    def __str__(self):
        repr_str = 'Volterra serie up to order {}:\n'.format(self.order_max)
        repr_str += self.__repr__()
        return repr_str

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