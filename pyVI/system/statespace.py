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
from colorama import Fore, Back, Style

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
    def _is_passive(self):
        raise NotImplementedError
            
    def __repr__(self):
        repr_str = ''
        for attribute in ['dim_input', 'dim_state', 'dim_output',
                          'Am', 'Bm', 'Cm', 'Dm', 'mpq', 'npq',
                          'linear']:
            repr_str += attribute + ' : ' + self.__dict__[attribute].__str__()
            repr_str += '\n'
        return repr_str
        
    def __str__(self):
        print_str = Back.RED + Fore.GREEN + Style.BRIGHT + \
                    'Space_state representation :' + Style.RESET_ALL + '\n'
        for name, matrice in iter([('State-to-state', self.Am),
                                   ('Input-to-state', self.Bm),
                                   ('State-to-output', self.Cm),
                                   ('Input-to-output', self.Dm)]):
            print_str += Fore.GREEN + Style.BRIGHT + name + ' :' + \
                         Style.RESET_ALL + '\n' + \
                         sp.pretty(matrice) + '\n'
        if not self.linear:
            print_str += Fore.GREEN + Style.BRIGHT + \
                         'List of non-null Mpq functions :' + \
                         Style.RESET_ALL + '\n' 
            for idx in iter(self.mpq.keys()):
                print_str += idx.__repr__() + ', '
            print_str += '\n' + Fore.GREEN + Style.BRIGHT + \
                         'List of non-null Npq functions :' + \
                         Style.RESET_ALL + '\n' 
            for idx in iter(self.npq.keys()):
                print_str += idx.__repr__() + ', '
        return print_str
    
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

