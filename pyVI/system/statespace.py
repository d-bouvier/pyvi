# -*- coding: utf-8 -*-
"""
Module for state-space representation of system.

This package creates classes that allows use of state-space representation for
linear and nonlinear systems (see
https://en.wikipedia.org/wiki/State-space_representation).

Class
-----
StateSpace :
    Defines physical systems by their state-space representations parameters.
NumericalStateSpace :
    Numerical version of StateSpace class.
SymbolicStateSpace :
    Characterize a system by its symbolic state-space representation.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 04 May 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

from sympy import pretty
from abc import abstractmethod
import warnings as warnings
from ..utilities.misc import Style
from ..simulation.tools import StateSpaceSimulationParameters
from ..simulation.simu import simulation as simulation_fct
from ..simulation.kernels import (time_kernel_computation,
                                  freq_kernel_computation,
                                  freq_kernel_computation_from_time_kernels)


#==============================================================================
# Class
#==============================================================================

class StateSpace:
    """
    Defines physical systems by their state-space representations parameters.

    This class represents a system (linear or nonlinear) by its state-space
    representation, a mathematical formalism used in control engineering
    (see https://en.wikipedia.org/wiki/State-space_representation).

    Attributes
    ----------
    A_m : array-like (numpy.ndarray or sympy.Matrix)
        State-to-state matrix
    B_m : array-like
        Input-to-state matrix
    C_m : array-like
        State-to-output matrix
    D_m : array-like
        Input-to-output matrix (feedtrhough matrix)
    mpq : dict {(int, int): tensor-like (numpy.ndarray or sympy.tensor.array)}
        Store multilinear Mpq functions (nonlinear terms of the state equation)
        in tensor forms.
    npq : dict {(int, int): tensor-like}
        Store multilinear Npq functions (nonlinear terms of the output equation)
        in tensor forms.
    dim : dict
        Dictionnaries with 3 entries giving respectively the dimension of:
        - the input
        - the state
        - the output
    pq_symmetry : boolean
        Indicates if multilinear Mpq and Npq tensors functions are symmetric.
    linear : boolean
        True if the system is linear.
    """

    def __init__(self, A_m, B_m, C_m, D_m, mpq_dict={}, npq_dict={},
                 pq_symmetry=False):
        """
        Initialisation function for System object.

        Parameters
        ----------
        A_m : array-like
            State-to-state matrix
        B_m : array-like
            Input-to-state matrix
        C_m : array-like
            State-to-output matrix
        D_m : array-like
            Input-to-output matrix (feedtrhough matrix)
        mpq_dict : dict {(int, int): tensor-like}
            Multilinear Mpq functions (nonlinear terms of the state equation)
            in tensor forms.
        npq_dict : dict {(int, int): tensor-like}
            Multilinear Npq functions (nonlinear terms of the output equation)
            in tensor forms.
        pq_symmetry : boolean, optional
            Indicates if multilinear Mpq and Npq tensors functions are
            symmetric.

        """

        # Initialize the linear part
        self.A_m = A_m
        self.B_m = B_m
        self.C_m = C_m
        self.D_m = D_m

        # Extrapolate system dimensions
        self.dim = {'input': B_m.shape[1],
                    'state': A_m.shape[0],
                    'output': C_m.shape[0]}

        # Initialize the nonlinear part
        self.mpq = mpq_dict
        self.npq = npq_dict
        self.pq_symmetry = pq_symmetry

        # Check dimensions and characteristics/categorization
        self._check_dim()
        self._ckeck_categories()

    def __repr__(self):
        """Lists all attributes and their values."""
        repr_str = ''
        # Print one attribute per line, in a alphabetical order
        for name in sorted(self.__dict__):
            repr_str += name + ' : ' + getattr(self, name).__str__() + '\n'
        return repr_str

    def __str__(self):
        """Prints the system's equation."""

        def list_nl_fct(dict_fct, name):
            temp_str = Style.RED + \
                       'List of non-zero {}pq functions'.format(name) + \
                       Style.RESET + '\n'
            for key in dict_fct.keys():
                temp_str += key.__repr__() + ', '
            temp_str = temp_str[0:-2] + '\n'
            return temp_str

        print_str = Style.UNDERLINE + Style.CYAN + Style.BRIGHT + \
                    'State-space representation :' + Style.RESET + '\n'
        for name, desc, mat in [ \
                    ('State {} A', 'state-to-state', self.A_m),
                    ('Input {} B', 'input-to-state', self.B_m),
                    ('Output {} C', 'state-to-output', self.C_m),
                    ('Feedthrough {} D', 'input-to-output', self.D_m)]:
            print_str += Style.GREEN + Style.BRIGHT + name.format('matrice') + \
                        ' (' + desc + ')' + Style.RESET + '\n' + \
                         pretty(mat) + '\n'
        if not self.linear:
            if len(self.mpq):
                print_str += list_nl_fct(self.mpq, 'M')
            if len(self.npq):
                print_str += list_nl_fct(self.npq, 'N')
        return print_str

    #=============================================#

    @abstractmethod
    def __add__(self, values_dict):
        raise NotImplementedError

    @abstractmethod
    def __radd__(self, values_dict):
        raise NotImplementedError

    #=============================================#

    def _check_dim(self):
        """Verify that input, state and output dimensions are respected."""
        # Check matrices shape
        self._check_dim_matrices()

        # Check that all nonlinear lambda functions works correctly
        for (p, q), mpq in self.mpq.items():
            self._check_dim_nl_tensor(p, q, mpq, 'M', self.dim['state'])
        for (p, q), npq in self.npq.items():
            self._check_dim_nl_tensor(p, q, npq, 'N', self.dim['output'])

        # If no error is raised
        self._is_single_input()
        self._is_single_output()
        self._dim_ok = True

        # Checking system type
        if self._single_input and self._single_output:
            self._type = 'SISO'
        elif self._single_input:
            self._type = 'SIMO'
        elif self._single_output:
            self._type = 'MISO'
        else:
            self._type = 'MIMO'

    def _check_dim_matrices(self):
        """Verify shape of the matrices used in the linear part."""
        def check_equal(iterator, value):
            return len(set(iterator)) == 1 and iterator[0] == value

        list_dim_state = [self.A_m.shape[0], self.A_m.shape[1],
                          self.B_m.shape[0], self.C_m.shape[1]]
        list_dim_input = [self.B_m.shape[1], self.D_m.shape[1]]
        list_dim_output = [self.C_m.shape[0], self.D_m.shape[0]]
        assert check_equal(list_dim_state, self.dim['state']), \
               'State dimension not consistent'
        assert check_equal(list_dim_input, self.dim['input']), \
               'Input dimension not consistent'
        assert check_equal(list_dim_output, self.dim['output']), \
               'Output dimension not consistent'

    def _check_dim_nl_tensor(self, p, q, tensor, name, dim_result):
        """Verify shape of the multilinear tensors."""
        str_tensor = '{}_{}{} tensor: '.format(name, p, q)
        shape = tensor.shape
        assert len(shape) == p + q + 1, \
               str_tensor + 'wrong number of dimension ' + \
               '(got {}, expected {}).'.format(len(shape), p + q + 1)
        assert shape[0] == dim_result, \
               str_tensor + 'wrong size for dimension 1 ' + \
               '(got {}, expected {}).'.format(dim_result, shape[0])
        for ind in range(p):
            assert shape[1+ind] == self.dim['state'], \
                   str_tensor + 'wrong size for dimension ' + \
                   '{} (got {}, expected {}).'.format(1+ind, shape[1+ind],
                                                      self.dim['state'])
        for ind in range(q):
            assert shape[1+p+ind] == self.dim['input'], \
                   str_tensor + 'wrong size for dimension ' + \
                   '{} (got {}, expected {}).'.format(1+p+ind, shape[1+p+ind],
                                                      self.dim['input'])

    def _is_single_input(self):
        """Check if the input dimension is one."""
        self._single_input = self.dim['input'] == 1
        # Warn that problems may occur if input dimension is not 1
        if not self._single_input:
            message = '\nInput dimension is not equal to 1' + \
                      ' (it is {}).\n'.format(self.dim['input']) + \
                      'Simulation, kernel computation, order separation and' + \
                      ' system  identification may not work as intended.\n'
            warnings.showwarning(message, UserWarning, __file__, 248, line='')

    def _is_single_output(self):
        """Check if the output dimension is one."""
        self._single_output = self.dim['output'] == 1
        # Warn that problems may occur if output dimension is not 1
        if not self._single_output:
            message = '\nOutput dimension is not equal to 1' + \
                      ' (it is {}).\n'.format(self.dim['output']) + \
                      'Simulation, kernel computation, order separation and' + \
                      ' system  identification may not work as intended.\n'
            warnings.showwarning(message, UserWarning, __file__, 258, line='')

    #=============================================#

    def _ckeck_categories(self):
        """Check in which categories the system belongs."""
        self._is_linear()
        self._is_dyn_eqn_linear_analytic()
        self._are_dynamical_nl_only_on_state()
        self._are_nl_colinear()

    def _is_linear(self):
        """Check if the system is linear."""
        self._state_eqn_linear = len(self.mpq) == 0
        self._output_eqn_linear = len(self.npq) == 0
        self.linear = self._state_eqn_linear and self._output_eqn_linear

    def _is_dyn_eqn_linear_analytic(self):
        """Check if the system dynamical equation is linear-analytic."""
        self.dyn_eqn_linear_analytic = True
        for p, q in self.mpq.keys():
            if q > 1:
                self.dyn_eqn_linear_analytic = False
                break

    def _are_dynamical_nl_only_on_state(self):
        self.dynamical_nl_only_on_state = 'unknown'

    def _are_nl_colinear(self):
        self.nl_colinear = 'unknown'


class NumericalStateSpace(StateSpace):
    """
    Numerical version of the StateSpace class.
    """
    #TODO docstring

    def simulation(self, input_signal, **options):
        #TODO docstring
        self._create_simulation_parameters(**options)
        return simulation_fct(input_signal, self.dim, self._simu.nl_order_max,
               self._simu.filter_mat, self.B_m, self.C_m, self.D_m,
               self.mpq, self.npq, self._simu.mpq_combinatoric,
               self._simu.npq_combinatoric, self._simu.holder_bias_mat)

    def compute_kernels(self, T, which='both', **options):
        #TODO docstring
        self._create_simulation_parameters(**options)

        if which == 'time':
            return self._compute_time_kernels(T)
        elif which == 'both':
            volterra_kernels = self._compute_time_kernels(T)
            transfer_kernels = \
                    self._compute_freq_kernels(T, time_kernels=volterra_kernels)
            return volterra_kernels, transfer_kernels
        elif which == 'freq':
            return self._compute_freq_kernels(T)

    #=============================================#

    def _create_simulation_parameters(self, **options):
        #TODO docstring
        self._simu = StateSpaceSimulationParameters(self.A_m, self.dim['state'],
                                                    self.mpq, self.npq,
                                                    self.pq_symmetry, **options)

    @abstractmethod
    def _compute_time_kernels(self, T):
        #TODO docstring
        return time_kernel_computation(T, self._simu.fs, self.dim,
                                       self._simu.nl_order_max,
                                       self._simu.filter_mat,
                                       self.B_m, self.C_m, self.D_m,
                                       self.mpq, self.npq,
                                       self._simu.mpq_combinatoric,
                                       self._simu.npq_combinatoric,
                                       self._simu.holder_bias_mat)

    def _compute_freq_kernels(self, T, time_kernels=None):
        #TODO docstring
        if time_kernels is not None:
            return freq_kernel_computation_from_time_kernels(time_kernels)
        else:
            return freq_kernel_computation(T, self._simu.fs, self.dim,
                                       self._simu.nl_order_max,
                                       self.A_m, self.B_m, self.C_m, self.D_m,
                                       self.mpq, self.npq,
                                       self._simu.mpq_combinatoric,
                                       self._simu.npq_combinatoric,
                                       self._simu.holder_order)

    @abstractmethod
    def convert2symbolic(self, values_dict):
        """Create a SymbolicStateSpace object of the system."""
        raise NotImplementedError

    @abstractmethod
    def symmetrization(self):
        #TODO docstring
        #TODO utiliser fct dans simulation/kernels
        raise NotImplementedError

class SymbolicStateSpace(StateSpace):
    """
    Symbolic version of the StateSpace class, using sympy package.

    In the future, this subclass will permit exportation to pdf via LaTex.
    """

    @abstractmethod
    def convert2numerical(self, values_dict):
        """Create a NumericalStateSpace object of the system."""
        raise NotImplementedError

    @abstractmethod
    def print2latex(self):
        """Create a LaTex document with the state-space representation."""
        raise NotImplementedError