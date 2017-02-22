# -*- coding: utf-8 -*-
"""
Module for handling sympy symbols.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 12 Sept. 2016
Developed for Python 3.5.1Uses:
 - sympy 1.0
"""

#==============================================================================
# Importations
#==============================================================================

import sympy as sp

#==============================================================================
# Class
#==============================================================================

class Symbols:
    _first_init = True
    _instance = None
    _list = ['t', 'tau', 's', 'w', 'f']
    
    def __new__(cls, order=0):
        if cls._first_init:
            cls._instance = object.__new__(cls)
        else:
            cls._instance._update(order)            
        return cls._instance 
            
    def __init__(self, order=0):
        if Symbols._first_init:
            self.order = order
            Symbols._first_init = False
            for var in self._list:
                str_symb = '{} {}(1:{})'.format(var, var, order+1)
                setattr(self, var, sp.symbols(str_symb, seq=True))

    def __repr__(self):
        repr_str = ''
        for var in self._list:
            repr_str += sp.pretty(getattr(self, var)) + '\n'
        return repr_str

    #=============================================#

    def _update(self, order):
        if order > self.order:
            for var in self._list:
                str_symb = '{}({}:{})'.format(var, self.order, order+1)
                setattr(self, var, getattr(self, var) + sp.symbols(str_symb,
                                                                   seq=True))            
            self.order = order

