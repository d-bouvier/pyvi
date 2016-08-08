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
from abc import abstractmethod

#==============================================================================
# Class
#==============================================================================

class Volterra:
    """ Class that defines Volterra serie of a system, described by:
    - n array of its kernels
    - its maximum order of nonlinearity
    Also defines if its kernels are symmetric and the list of sympy symbols used 
    in the expressions."""
    
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
        for idx, kernel in enumerate(self.kernels):
            self.list_kernels[idx] = kernel.symmetrize()

    @abstractmethod
    def regularize(self):
        for idx, kernel in enumerate(self.kernels):
            self.list_kernels[idx] = kernel.regularize()

    @abstractmethod
    def triangularize(self):
        for idx, kernel in enumerate(self.kernels):
            self.list_kernels[idx] = kernel.triangularize()

    @abstractmethod
    def inverse(self):
        return NotImplementedError
        
    @abstractmethod
    def composition(self, volterra):
        return NotImplementedError


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
        else:
            self.symmetric = None
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
        
    def __str__(self):
        repr_str = 'Volterra kernel of order {}:\n'.format(self.order)
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