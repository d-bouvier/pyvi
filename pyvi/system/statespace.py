# -*- coding: utf-8 -*-
"""
Module for state-space representation of system.

This package creates classes that allows use of state-space representation for
linear and nonlinear systems (see
https://en.wikipedia.org/wiki/State-space_representation).

Class
-----
StateSpace :
    Defines physical systems by their state-space representations.
NumericalStateSpace :
    Numerical version of StateSpace class.
SymbolicStateSpace :
    Symbolic version of StateSpace class, using sympy package.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 24 Nov. 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import warnings
from abc import abstractmethod
from operator import itemgetter
import sympy as sp
import numpy as np
import scipy.linalg as sc_lin
from ..utilities.misc import Style


#==============================================================================
# Class
#==============================================================================

class StateSpace:
    """
    Defines physical systems by their state-space representations.

    This class represents a system (linear or nonlinear) by its state-space
    representation, a mathematical formalism used in control engineering
    (see https://en.wikipedia.org/wiki/State-space_representation).


    Parameters
    ----------
    A_m : array-like (numpy.ndarray or sympy.Matrix)
        State-to-state matrix.
    B_m : array-like
        Input-to-state matrix.
    C_m : array-like
        State-to-output matrix.
    D_m : array-like
        Input-to-output matrix (feedtrhough matrix).
    mpq : dict((int, int): tensor-like (numpy.ndarray or sympy.tensor.array))
        Multilinear Mpq functions (nonlinear terms of the state equation)
        in tensor forms.
    npq : dict((int, int): tensor-like)
        Multilinear Npq functions (nonlinear terms of the output equation)
        in tensor forms.
    pq_symmetry : boolean, optional (default=False)
        Indicates if multilinear Mpq and Npq tensors functions are
        symmetric.

    Attributes
    ----------
    A_m, B_m, C_m, D_m : array_like
    mpq, npq : dict((int, int): tensor-like)
    pq_symmetry : boolean
    dim : dict(str: int)
        Dictionnaries with 3 entries giving the dimension of the input, the
        state and the output.
    linear : boolean
        True if the system is linear.
    state_eqn_linear_analytic : boolean
        True if the system is linear-analytic (q<2 for every Mpq and Npq).
    dynamical_nl_only_on_state : boolean
        True if the dynamical equation's nonlinearities are only on the state.
    _dim_ok : boolean
        True if dimensions all array and tensor dimensions corresponds.
    _type : {'SISO', 'SIMO', 'MISO', 'MIMO'}
        System's type (in terms of number of inputs and outputs).
    _single_input : boolean
        True if the system takes unidimensional signals as inputs.
    _single_output : boolean
        True if the system outputs unidimensional signals.
    """

    def __init__(self, A_m, B_m, C_m, D_m, mpq={}, npq={}, pq_symmetry=False):

        # Initialize the linear part
        self.A_m = A_m
        self.B_m = B_m
        self.C_m = C_m
        self.D_m = D_m

        # Extrapolate system dimensions
        self._system_dimension()

        # Initialize the nonlinear part
        self.mpq = mpq
        self.npq = npq
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
        for name, desc, mat in [
                    ('State {} A', 'state-to-state', self.A_m),
                    ('Input {} B', 'input-to-state', self.B_m),
                    ('Output {} C', 'state-to-output', self.C_m),
                    ('Feedthrough {} D', 'input-to-output', self.D_m)]:
            print_str += Style.GREEN + Style.BRIGHT + \
                         name.format('matrice') + ' (' + desc + ')' + \
                         Style.RESET + '\n' + sp.pretty(mat) + '\n'
        if not self.linear:
            if self.mpq:
                print_str += list_nl_fct(self.mpq, 'M')
            if self.npq:
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

    def _system_dimension(self):
        """Get dimensions of the system from matrices of the linear part."""

        # Initialization
        self.dim = dict()

        # Dimension of state
        if len(self.A_m.shape) != 2:
            raise ValueError("State-to-state matrix 'A_m' is not a 2D-array " +
                             "(it has", len(self.A_m.shape), "dimensions).")
        if self.A_m.shape[0] != self.A_m.shape[1]:
            raise ValueError("State-to-state matrix 'A_m' is not square " +
                             "(it has shape {}).".format(self.A_m.shape))
        self.dim['state'] = self.A_m.shape[0]

        # Dimension of input
        if not len(self.B_m.shape) in [1, 2]:
            raise ValueError("Input-to-state matrix 'B_m' is not a 1D or " +
                             "2D-array (it has", len(self.B_m.shape),
                             "dimensions).")
        if self.B_m.shape[0] != self.dim['state']:
            raise ValueError("Shape of input-to-state matrix 'B_m'",
                             self.B_m.shape, "is not consistent with the " +
                             "state's dimension {}.".format(self.dim['state']))
        if len(self.B_m.shape) == 1:
            self.B_m.shape = (self.dim['state'], 1)
        self.dim['input'] = self.B_m.shape[1]

        # Dimension of output
        C_shape = self.C_m.shape

        if not len(C_shape) in [1, 2]:
            raise ValueError("State-to-output matrix 'C_m' is not a 1D or " +
                             "2D-array (it has", len(self.B_m.shape),
                             "dimensions).")
        if C_shape[-1] != self.dim['state']:
            raise ValueError("Shape of state-to-output matrix 'C_m'",
                             C_shape, "is not consistent with the state's" +
                             "dimension {}.".format(self.dim['state']))
        if len(C_shape) == 1:
            C_shape = (1, self.dim['state'])
        self.dim['output'] = C_shape[0]

        if self.D_m.shape != (self.dim['output'], self.dim['input']):
            try:
                self.D_m.shape = (self.dim['output'], self.dim['input'])
            except ValueError:
                raise ValueError("Shape of input-to-output matrix 'D_m'",
                                 self.D_m.shape, "is not consistent with the" +
                                 " input and/or output's dimension " +
                                 "(respectively", self.dim['input'], "and",
                                 self.dim['output'])
            except AttributeError:
                raise AttributeError("Shape of input-to-output matrix 'D_m'",
                                     self.D_m.shape, "cannot be casted to",
                                     "({},".format(self.dim['input']),
                                     "{}) because".format(self.dim['output']),
                                     "shape's of sympy Matrix are not " +
                                     "alterable.")

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
        if not check_equal(list_dim_state, self.dim['state']):
            raise ValueError('State dimension not consistent')
        if not check_equal(list_dim_input, self.dim['input']):
            raise ValueError('Input dimension not consistent')
        if not check_equal(list_dim_output, self.dim['output']):
            raise ValueError('Output dimension not consistent')

    def _check_dim_nl_tensor(self, p, q, tensor, name, dim_result):
        """Verify shape of the multilinear tensors."""
        str_tensor = '{}_{}{} tensor: '.format(name, p, q)
        shape = tensor.shape

        if len(shape) != p + q + 1:
            raise ValueError(str_tensor + 'wrong number of dimension',
                             '(got {},'.format(len(shape)),
                             'expected {}).'.format(p + q + 1))
        if shape[0] != dim_result:
            raise ValueError(str_tensor + 'wrong size for dimension 1',
                             '(got {},'.format(dim_result),
                             'expected {}).'.format(shape[0]))
        for ind in range(p):
            if shape[1+ind] != self.dim['state']:
                raise ValueError(str_tensor + 'wrong size for dimension',
                                 '{} (got {},'.format(1+ind, shape[1+ind]),
                                 'expected {}).'.format(self.dim['state']))
        for ind in range(q):
            if shape[1+p+ind] != self.dim['input']:
                raise ValueError(str_tensor + 'wrong size for dimension',
                                 '{} (got {},'.format(1+p+ind, shape[1+p+ind]),
                                 'expected {}).'.format(self.dim['input']))

    def _is_single_input(self):
        """Check if the input dimension is one."""
        self._single_input = self.dim['input'] == 1
        # Warn that problems may occur if input dimension is not 1
        if not self._single_input:
            message = '\nInput dimension is not equal to 1' + \
                      ' (it is {}).\n'.format(self.dim['input']) + \
                      'Simulation, kernel computation, order separation ' + \
                      'and system identification may not work as intended.'
            warnings.warn(message, UserWarning)

    def _is_single_output(self):
        """Check if the output dimension is one."""
        self._single_output = self.dim['output'] == 1
        # Warn that problems may occur if output dimension is not 1
        if not self._single_output:
            message = '\nOutput dimension is not equal to 1' + \
                      ' (it is {}).\n'.format(self.dim['output']) + \
                      'Simulation, kernel computation, order separation ' + \
                      'and system identification may not work as intended.'
            warnings.warn(message, UserWarning)

    #=============================================#

    def _ckeck_categories(self):
        """Check in which categories the system belongs."""
        self._dynamical_equation_category()
        self._output_equation_category()
        self.linear = self._dyn_eqn_linear & self._out_eqn_linear

    def _dynamical_equation_category(self):
        """Check the category of the dynamical equation."""
        pq_list = list(self.mpq.keys())
        if len(pq_list) == 0:
            p_max = 0
            q_max = 0
        else:
            p_max = max(pq_list, key=itemgetter(0))[0]
            q_max = max(pq_list, key=itemgetter(1))[1]
        self._dyn_eqn_input_affine = q_max < 2
        self._dyn_eqn_state_affine = p_max < 2
        self._dyn_nl_only_on_input = p_max == 0
        self._dyn_nl_only_on_state = q_max == 0
        self._dyn_eqn_linear = (p_max == 0) & (q_max == 0)
        self._dyn_eqn_bilinear = \
            self._dyn_eqn_input_affine & self._dyn_eqn_state_affine

    def _output_equation_category(self):
        """Check the category of the output equation."""
        pq_list = list(self.npq.keys())
        if len(pq_list) == 0:
            p_max = 0
            q_max = 0
        else:
            p_max = max(pq_list, key=itemgetter(0))[0]
            q_max = max(pq_list, key=itemgetter(1))[1]
        self._out_eqn_input_affine = q_max < 2
        self._out_eqn_state_affine = p_max < 2
        self._out_nl_only_on_input = p_max == 0
        self._out_nl_only_on_state = q_max == 0
        self._out_eqn_linear = (p_max == 0) & (q_max == 0)
        self._out_eqn_bilinear = \
            self._out_eqn_input_affine & self._out_eqn_state_affine


class NumericalStateSpace(StateSpace):
    """
    Numerical version of the StateSpace class.

    Parameters
    ----------
    A_m : array-like (numpy.ndarray)
    B_m : array-like
    C_m : array-like
    D_m : array-like
    mpq : dict((int, int): tensor-like (numpy.ndarray))
    npq : dict((int, int): tensor-like)
    pq_symmetry : boolean, optional (default=False)

    Attributes
    ----------
    A_m, B_m, C_m, D_m : array_like
    mpq, npq : dict((int, int): tensor-like)
    pq_symmetry : boolean
    dim : dict(str: int)
    linear : boolean
    state_eqn_linear_analytic : boolean
    dynamical_nl_only_on_state : boolean
    nl_colinear : boolean
        True if dynamical nonlinearities are colinear to the input-to-state
        matrix.

    Methods
    -------
    convert2symbolic(values_dict)
        Returns a SymbolicStateSpace object of the system.

    See also
    --------
    StateSpace : Parent class.
    """

    def _ckeck_categories(self):
        """Check in which categories the system belongs."""
        StateSpace._ckeck_categories(self)
        self._are_dyn_nl_colinear()

    def _are_dyn_nl_colinear(self):
        """Check colinearity of dynamical nonlinearities and input-to-state."""
        self._nl_colinear = True
        norm_B = sc_lin.norm(self.B_m, ord=2)
        for (p, q), mpq in self.mpq.items():
            temp = np.tensordot(self.B_m.transpose(), mpq, 1).squeeze()
            norm_mpq = sc_lin.norm(mpq, ord=2, axis=0).squeeze()
            if not np.allclose(temp, norm_B*norm_mpq):
                self._nl_colinear = False
                break

    @abstractmethod
    def convert2symbolic(self, values_dict):
        """Returns a SymbolicStateSpace object of the system."""
        raise NotImplementedError


class SymbolicStateSpace(StateSpace):
    """
    Symbolic version of the StateSpace class, using sympy package.

    In the future, this subclass will permit exportation to pdf via LaTex.

    Parameters
    ----------
    A_m : array-like (sympy.Matrix)
    B_m : array-like
    C_m : array-like
    D_m : array-like
    mpq : dict((int, int): tensor-like (sympy.tensor.array))
    npq : dict((int, int): tensor-like)
    pq_symmetry : boolean, optional (default=False)

    Attributes
    ----------
    A_m, B_m, C_m, D_m : array_like
    mpq, npq : dict((int, int): tensor-like)
    pq_symmetry : boolean
    dim : dict(str: int)
    linear : boolean
    state_eqn_linear_analytic : boolean
    dynamical_nl_only_on_state : boolean

    Methods
    -------
    convert2numerical(values_dict)
        Returns a NumericalStateSpace object of the system.
    print2latex()
        Creates a LaTex document with the state-space representation.

    See also
    --------
    StateSpace : Parent class.
    """

    @abstractmethod
    def convert2numerical(self, values_dict):
        """Returns a NumericalStateSpace object of the system."""
        raise NotImplementedError

    @abstractmethod
    def print2latex(self):
        """Creates a LaTex document with the state-space representation."""
        raise NotImplementedError
