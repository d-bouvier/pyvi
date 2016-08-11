# -*- coding: utf-8 -*-
"""
Description

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Created on Thu Aug 11 10:16:25 2016
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
            print('Je ne pass pas par la')
            self.order = order
            Symbols._first_init = False
            for var in self._list:
                setattr(self, var,
                        sp.symbols('{} {}(1:{})'.format(var, var, order+1)))

    def _update(self, n):
        if n > self.order:
            self.order = n
            for var in self._list:
                setattr(self, var, getattr(self, var) + \
                        sp.symbols('{}({}:{})'.format(var, self.order, n+1)))            

    def __repr__(self):
        repr_str = ''
        for var in self._list:
            repr_str += sp.pretty(getattr(self, var)) + '\n'
        return repr_str

