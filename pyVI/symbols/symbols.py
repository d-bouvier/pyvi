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
    _exists = False
    
    def __new__(cls, order):
        if cls._exists:
            cls._instance._update(order)
        else:
            cls._instance = object.__new__(cls)
            cls._exists = True
        return cls._instance 
        
    def __init__(self, n):
        self.order = n
        for var in ['t', 'tau', 's', 'w', 'f']:
            setattr(self, var, sp.symbols('{} {}(1:{})'.format(var, var, n+1)))
    
    def _update(self, n):
        if n > self.order:
            self.order = n
            for var in ['t', 'tau', 's', 'w', 'f']:
                setattr(self, var, getattr(self, var) + \
                        sp.symbols('{}({}:{})'.format(var, self.order, n+1)))

    def __repr__(self):
        repr_str = ''
        for var in ['t', 'tau', 's', 'w', 'f']:
            repr_str += sp.pretty(getattr(self, var)) + '\n'
        return repr_str

