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

class Kernel:
    """ Class that defines a Volterra kernel described by:
    - its expression
    - its order of nonlinearity
    Also defines if it's a symmetric kernel and the list of sympy symbols used 
    in the expression."""
    
    def __init__(self, expr=sp.zeros(1), order=1, **kwargs):
        self.expr = expr
        self.order = order
        self.dim_input = self.expr.shape[1]
        self.dim_output = self.expr.shape[0]
        
        self.symmetric = kwargs.get('symmetric', None)
        self.symb_name = kwargs.get('name',
                                    sp.Function('H{}'.format(self.order)))
        self.symb_var = kwargs.get('var', None)
        
        self.a = self.__dict__.__str__()
    
    def __repr__(self):
        return sp.pretty(self.expr)
        
    def __str__(self):
        repr_str = 'Volterra kernel of order {}:\n'.format(self.order)
        repr_str += sp.pretty(self.symb_name(*self.symb_var))
        repr_str += ' = '
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


class Volterra:
    """ Class that defines Volterra serie of a system, described by:
    - n array of its kernels
    - its maximum order of nonlinearity
    Also defines if its kernels are symmetric."""
    
    def __init__(self, kernels=[Kernel()], **kwargs):
        self.kernels = kernels
        self.list_kernels = []
        for idx, kernel in enumerate(self.kernels):
            self.list_kernels.append(kernel.order)
        
        self.order_max = kwargs.get('order_max', max(self.list_kernels))
        self.symmetric = kwargs.get('symmetric')
        if self.symmetric is None:
            self.symmetric = True
            for idx, kernel in enumerate(self.kernels):
                self.symmetric = self.symmetric and kernel.symmetric

        self.a = self.__dict__.__str__()
    
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


#==============================================================================
# Functions
#==============================================================================