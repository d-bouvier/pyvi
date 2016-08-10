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
    """Characterize a system by its state-space representation.
    
    This class represents a system (linear or nonlinear) by its state-space
    representation, a mathematical representation used in control engineering
    (see https://en.wikipedia.org/wiki/State-space_representation).
    
    Relies on the Sympy module.
    
    Attributes
    ----------
    dim_input: int
        Dimension of the input vector
    dim_state: int
        Dimension of the state vector
    dim_output: int
        Dimension of the output vector
    Am : sympy.Matrix
        The 'state (or system) matrix' (i.e. state-to-state matrix)
    Bm : sympy.Matrix
        The 'input matrix' (i.e. input-to-state matrix)
    Cm : sympy.Matrix
        The 'output matrix' (i.e. state-to-output matrix)
    Dm : sympy.Matrix
        The 'feedtrough matrix' (i.e. input-to-output matrix)
    mpq_dict : dict of {(int, int): lambda}
        Dictionnary of lambda functions describing the nonlinear part of the
        multivariate Taylor series expansion of the state equation.
    npq_dict : dict of {(int, int): lambda}
        Dictionnary of lambda functions describing the nonlinear part of the
        multivariate Taylor series expansion of the output equation.
    linear : boolean
        Tells if the system is linear.
    """
    
    
    def __init__(self, Am, Bm, Cm, Dm, mpq_dict={}, npq_dict={}, **kwargs):
        """Initialize the representation of the system.
        
        Mandatory parameters
        --------------------
        Am : sympy.Matrix
            The 'state (or system) matrix' (i.e. state-to-state matrix)
        Bm : sympy.Matrix
            The 'input matrix' (i.e. input-to-state matrix)
        Cm : sympy.Matrix
            The 'output matrix' (i.e. state-to-output matrix)
        Dm : sympy.Matrix
            The 'feedtrough matrix' (i.e. input-to-output matrix)
        
        Optional parameters
        -------------------
        mpq_dict : dict of {(int, int): lambda}
            Each lambda function represents a multilinear M_pq function,
            characterized by its key (p, q), that represents a nonlinear part
            of the state equation. It should take p + q input
            arguments (sympy.Matrix of shape (self.dim_state, 1) for the first
            p and (self.dim_input, 1) for the last q), and should output a
            sympy.Matrix of shape (self.sim_state, 1).
        npq_dict : dict of {(int, int): lambda}
            Each lambda function represents a multilinear N_pq function,
            characterized by its key (p, q), that represents a nonlinear part
            of the output equation. It should take p + q input
            arguments (sympy.Matrix of shape (self.dim_state, 1) for the first
            p and (self.dim_input, 1) for the last q), and should output a
            sympy.Matrix of shape (self.dim_output, 1).
        
        """
        
        # Initialize the linear part
        self.Am = Am        
        self.Bm = Bm
        self.Cm = Cm
        self.Dm = Dm        

        # Extrapolate system dimensions
        self.dim_state = Am.shape[0]
        self.dim_input = Bm.shape[1]
        self.dim_output = Cm.shape[0]
        
        # Initialize the nonlinear part
        self.mpq = mpq_dict
        self.npq = npq_dict
        
        # CHeck dimension and linearity
        self._dim_ok = self._check_dim()
        self.linear = self._is_linear()
        

    def _check_dim(self):
        """Verify that input, state and output dimensions are respected."""
        # Check matrices shape
        self._check_dim_matrices()
        
        # Check that all nonlinear lambda functions works correctly
        for (p, q), fct in self.mpq.items():
            self._check_dim_nl_fct(p, q, fct, 'M', self.dim_state)
        for (p, q), fct in self.npq.items():
            self._check_dim_nl_fct(p, q, fct, 'N', self.dim_output)
        # If no error is raised, return True
        return True


    def _check_dim_matrices(self):
        """Verify shape of the matrices used in the linear part.""" 
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


    def _check_dim_nl_fct(self, p, q, fct, name, dim_result):
        """Verify shape and functionnality of the multilinear functions."""
        # Check that each nonlinear lambda functions:
        if fct.__code__.co_argcount != p + q:
            # - accepts the good number of input arguments
            raise NameError('{}_{}{} function: '.format(name, p, q) + \
                            'wrong number of input arguments ' + \
                            'got {}, '.format(fct.__code__.co_argcount) + \
                            'expected {}).'.format(p + q))
        else:
            try:
                state_vectors = (sp.ones(self.dim_state),)*p
                input_vectors = (sp.ones(self.dim_input),)*q
                result_vector = fct(*state_vectors, *input_vectors)
            except IndexError:
                # - accepts vectors of appropriate shapes
                raise IndexError('{}_{}{} function: '.format(name, p, q) + \
                                 'some index exceeds dimension of ' + \
                                 'input and/or state vectors.')
            except:
                # - does not cause error
                raise NameError('{}_{}{} function: '.format(name, p, q) + \
                                 'creates a {}.'.format(sys.exc_info()[0]))
            if result_vector.shape != (dim_result, 1):
                # - returns a vector of appropriate shape
                raise NameError('{}_{}{} function: '.format(name, p, q) + \
                                'wrong shape for the output ' + \
                                '(got {}, '.format(result_vector.shape) + \
                                'expected {}).'.format((dim_result,1)))


    def _is_linear(self):
        """Check if the system is linear."""
        return len(self.mpq) == 0 and len(self.npq) == 0


    @abstractmethod
    def _is_passive(self):
        """Check if the system is passive."""
        raise NotImplementedError

            
    def __repr__(self):
        """Lists all attributes and their values."""
        repr_str = ''
        # Print one attribute per line, in a defined order
        for attribute in ['dim_input', 'dim_state', 'dim_output',
                          'Am', 'Bm', 'Cm', 'Dm', 'mpq', 'npq',
                          'linear', '_dim_ok']:
            repr_str += attribute + ' : ' + self.__dict__[attribute].__str__()
            repr_str += '\n'
        return repr_str

        
    def __str__(self):
        """Prints the system's equation."""
        # Not yet implemented as wanted
        print_str = '\033[4m' + Fore.RED + Style.BRIGHT + \
                    'State-space representation :' + Style.RESET_ALL + '\n'
        for name, matrice in iter([('State-to-state', self.Am),
                                   ('Input-to-state', self.Bm),
                                   ('State-to-output', self.Cm),
                                   ('Input-to-output', self.Dm)]):
            print_str += Fore.GREEN + Style.BRIGHT + name + ' matrice' + \
                         Style.RESET_ALL + '\n' + \
                         sp.pretty(matrice) + '\n'
        if not self.linear:
            print_str += Fore.GREEN + Style.BRIGHT + \
                         'List of non-null Mpq functions' + \
                         Style.RESET_ALL + '\n' 
            for idx in iter(self.mpq.keys()):
                print_str += idx.__repr__() + ', '
            print_str += '\n' + Fore.GREEN + Style.BRIGHT + \
                         'List of non-null Npq functions' + \
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

