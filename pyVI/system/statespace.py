# -*- coding: utf-8 -*-
"""
Module for state-space representation.

This package creates classes that allows use of state-space
representation for linear and nonlinear systems (see
https://en.wikipedia.org/wiki/State-space_representation).

Class
-----
StateSpace :
    Defines physical systems by their state-space representations parameters.
SymbolicStateSpace :
    Characterize a system by its symbolic state-space representation.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 3 Nov. 2016
Developed for Python 3.5.1
"""

#==============================================================================
# Importations
#==============================================================================

from pyvi.tools.utilities import Style
from abc import abstractmethod
import sys as sys
import sympy as sp


#==============================================================================
# Class
#==============================================================================

class StateSpace:
    """Defines physical systems by their state-space representations parameters.

    Attributes
    ----------
    A_m : numpy.ndarray
        State-to-state matrix
    B_m : numpy.ndarray
        Input-to-state matrix
    C_m : numpy.ndarray
        State-to-output matrix
    D_m : numpy.ndarray
        Input-to-output matrix (feedtrhough matrix)
    dim : dict
        Dictionnaries with 3 entries giving respectively the dimension of:
        - the input
        - the state
        - the output
    is_mpq_used, is_npq_used : function (int, int: boolean)
        Indicates, for given p & q, if the Mpq and Npq function is used in the
        system.
    mpq, npq : dict
        Store multilinear Mpq and Npq functions, in one of the two following
        forms:
        - numpy.ndarray in 'tensor' mode;
        - function (int, numpy.ndarray, ..., numpy.ndarray: numpy.ndarray) in
        'function' mode.
    sym_bool : boolean
        Indicates if multilinear Mpq and Npq functions are symmetric.
    mode : {'tensor', 'function'}
        Define in which mode multilinear Mpq and Npq functions are stored.
    """

    def __init__(self, A_m, B_m, C_m, D_m,
                 h_mpq_bool, h_npq_bool, mpq_dict, npq_dict,
                 sym_bool=False, mode='tensor'):
        """
        Initialisation function for System object.

        Parameters
        ----------
        A_m : numpy.ndarray
            State-to-state matrix
        B_m : numpy.ndarray
            Input-to-state matrix
        C_m : numpy.ndarray
            State-to-output matrix
        D_m : numpy.ndarray
            Input-to-output matrix (feedtrhough matrix)
        h_mpq_bool, npq : function (int, int: boolean)
            Indicates, for given p & q, if the Mpq and Npq function is used in
            the system.
        mpq_dict, npq_dict : dict
            Store multilinear Mpq and Npq functions, in one of the two following
            forms:
            - numpy.ndarray in 'tensor' mode;
            - function (int, numpy.ndarray, ..., numpy.ndarray: numpy.ndarray)
            in 'function' mode.
        sym_bool : boolean, optional
            Indicates if multilinear Mpq and Npq functions are symmetric.
        mode : {'tensor', 'function'}, optional
            Define in which mode multilinear Mpq and Npq functions are stored.

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
        self.is_mpq_used = h_mpq_bool
        self.is_npq_used = h_npq_bool
        self.mpq = mpq_dict
        self.npq = npq_dict

        self.sym_bool = sym_bool
        self.mode = mode


class SymbolicStateSpace:
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
            temp_str = Style.PURPLE + \
                       'List of non-zero {}pq functions'.format(name) + \
                       Style.RESET + '\n'
            for key in dict_fct.keys():
                temp_str += key.__repr__() + ', '
            temp_str = temp_str[0:-2] + '\n'
            return temp_str

        # Not yet implemented as wanted
        print_str = Style.UNDERLINE + Style.BLUE + Style.BRIGHT + \
                    'State-space representation :' + Style.RESET + '\n'
        for name, desc, mat in [ \
                    ('State {} A', 'state-to-state', self.Am),
                    ('Input {} B', 'input-to-state', self.Bm),
                    ('Output {} C', 'state-to-output', self.Cm),
                    ('Feedthrough {} D', 'input-to-output', self.Dm)]:
            print_str += Style.BLUE + Style.BRIGHT + name.format('matrice') + \
                        ' (' + desc + ')' + Style.RESET + '\n' + \
                         sp.pretty(mat) + '\n'
        if not self.linear:
            if len(self.mpq):
                print_str += list_nl_fct(self.mpq, 'M')
            if len(self.npq):
                print_str += list_nl_fct(self.npq, 'N')
        return print_str

    #=============================================#

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
        assert check_equal(list_dim_state, self.dim_state), \
               'State dimension not consistent'
        assert check_equal(list_dim_input, self.dim_input), \
               'Input dimension not consistent'
        assert check_equal(list_dim_output, self.dim_output), \
               'Output dimension not consistent'


    def _check_dim_nl_fct(self, p, q, fct, name, dim_result):
        """Verify shape and functionnality of the multilinear functions."""
        str_fct = '{}_{}{} function: '.format(name, p, q)
        # Check that each nonlinear lambda functions:
        # - accepts the good number of input arguments
        assert fct.__code__.co_argcount == p + q, \
               str_fct + 'wrong number of input arguments ' + \
               '(got {}, expected {}).'.format(fct.__code__.co_argcount, p + q)
        try:
            state_vectors = (sp.ones(self.dim_state),)*p
            input_vectors = (sp.ones(self.dim_input),)*q
            result_vector = fct(*state_vectors, *input_vectors)
        # - accepts vectors of appropriate shapes
        except IndexError:
            raise IndexError(str_fct + 'some index exceeds dimension of ' + \
                             'input and/or state vectors.')
        # - does not cause error
        except:
            raise NameError(str_fct + 'creates a ' + \
                            '{}.'.format(sys.exc_info()[0]))
        # - returns a vector of appropriate shape
        assert result_vector.shape == (dim_result, 1), \
               str_fct + 'wrong shape for the output (got ' + \
               '{}, expected {}).'.format(result_vector.shape, (dim_result,1))


    def _is_linear(self):
        """Check if the system is linear."""
        return len(self.mpq) == 0 and len(self.npq) == 0


    @abstractmethod
    def _is_passive(self):
        """Check if the system is passive."""
        raise NotImplementedError

    #=============================================#

    @abstractmethod
    def print2latex(self):
        """Create a LaTex document with the state-space representation."""
        raise NotImplementedError


    def compute_linear_filter(self):
        """Compute the multi-dimensional filter of the system."""
        self.W_filter = Filter(self.Am, self.dim_state)


    @abstractmethod
    def simulation(self):
        """Compute the output of the system for a given input."""
        raise NotImplementedError



class Filter:
    """Multidimensional filter of a system in its state-space representation."""

    def __init__(self, Am, state_size):
        from symbols.symbols import Symbols
        self.symb_var = Symbols(1).s[0]
        temp_mat = self.symb_var * sp.eye(state_size) - Am
        self.expr = temp_mat.inv()
        self.common_den = temp_mat.det()
        self.mat = sp.simplify(self.expr * self.common_den)


    def __str__(self):
        expr = sp.Mul(self.mat, sp.Pow(self.common_den, sp.Integer(-1)),
                      evaluate=False)
        print_str = '\n' + sp.pretty( expr )
        return print_str

def loudspeaker_sica(version='tristan', output='pos', mode='tensor'):
    """
    Function that create and returns the System object corresponding to the
    SICA Z000900 loudspeaker
    (http://www.sica.it/media/Z000900C.pdf551d31b7b491e.pdf).

    Parameters
    ----------
    version : {'tristan', 'CFA'}, optional
        Version to simulate.
    output : {'pos', 'current'}, optional
        Defines the output of the system
    mode : {'tensor', 'function'}, optional
        Mode in which are stored Mpq and Npq multilinear functions

    Returns
    -------
    Object of class System.

    """

    import numpy as np

    ## Physical parameters ##
    # Electric parameters
    if version == 'tristan': # Electodynamic driving parameter [T.m]
        Bl = 2.9
    elif version == 'CFA':
        Bl = 2.99
    Re = 5.7 # Electrical resistance of voice coil   [Ohm]
    Le = 0.11e-3 # Coil inductance [H]
    # Mechanical parameters
    Mms = 1.9e-3; # Mechanical mass [kg]
    if version == 'tristan':
        Rms = 0.406 # Mechanical damping and drag force [kg.s-1]
        k = [912.2789, 611.4570, 8e07] # Suspension stiffness [N.m-1]
    elif version == 'CFA':
        Cms = 544e-6; # Mechanical compliance [m.N-1]
        Qms = 4.6;
        k = [1/Cms, -554420.0, 989026000] # Suspension stiffness [N.m-1]
        # Mechanical damping and drag force [kg.s-1]
        Rms = np.sqrt(k[0] * Mms)/Qms;

    # State-space matrices
    A_m = np.array([[-Re/Le, 0, -Bl/Le],
                    [0, 0, 1],
                    [Bl/Mms, -k[0]/Mms, -Rms/Mms]]) # State-to-state matrix
    B_m = np.array([[1/Le], [0], [0]]); # Input-to-state matrix
    # State-to-output matrix
    if output == 'pos':
        C_m = np.array([[0, 1, 0]])
    elif output == 'current':
        C_m = np.array([[1, 0, 0]])
    D_m = np.zeros((1, 1)) # Input-to-output matrix

    # Handles for fonction saying if Mpq and Npq functions are used
    h_mpq_bool = (lambda p, q: (p<=3) & (q==0))
    h_npq_bool = (lambda p, q: False)

     # Dictionnaries of Mpq & Npq tensors
    if mode == 'tensor':
        m20 = np.zeros((3, 3, 3))
        m20[2, 1, 1] = -k[1]/Mms
        m30 = np.zeros((3, 3, 3, 3))
        m30[2, 1, 1, 1] = -k[2]/Mms
    elif mode == 'function':
        m20 = lambda a, x1, x2: np.stack((np.zeros(a), np.zeros(a), \
                                    -k[1]/Mms * x1[1] * x2[1]), axis=0)
        m30 = lambda a, x1, x2, x3: np.stack((np.zeros(a), np.zeros(a), \
                                    -k[2]/Mms * x1[1] * x2[1] * x3[1]), axis=0)

    mpq_dict = {(2, 0): m20, (3, 0): m30}
    npq_dict = dict()

    return StateSpace(A_m, B_m, C_m, D_m, h_mpq_bool, h_npq_bool,
                      mpq_dict, npq_dict, sym_bool=True, mode=mode)

def simple_system():
    """
    Function that create and returns the System object corresponding to a
    simple system for simulation test.

    Returns
    -------
    Object of class System.

    """

    import numpy as np

    m20 = np.zeros((2, 2, 2))
    m20[1, 0, 0] = 1
    m10 = np.zeros((2, 2, 1))
    m10[0, 1, 0] = -1
    m02 = np.zeros((2, 1, 1))
    m02[0, 0, 0] = 2

    return StateSpace(np.array([[-1, 0], [1/2, 1/2]]), np.array([[1], [0]]),
                      np.array([[1, 0]]), np.zeros((1, 1)),
                      (lambda p, q: (p+q)<3), (lambda p, q: False),
                      {(2, 0): m20, (1, 1): m10, (0, 2): m02}, dict(),
                      sym_bool=True, mode='tensor')
