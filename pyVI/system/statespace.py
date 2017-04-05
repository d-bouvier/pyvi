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
Uses:
 - numpy 1.11.1
 - sympy 1.0
 - scipy 0.18.0
 - matplotlib 1.5.1
 - pyvi 0.1
 - itertools
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
import sympy as sp
from pyvi.tools.utilities import Style
from abc import abstractmethod
import sys as sys
import itertools as itr


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

        # Check dimension and linearity
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
        for (p, q), mpq in self.mpq.items():
            if self.mode == 'function':
                self._check_dim_nl_fct(p, q, mpq, 'M', self.dim['state'])
            else:
                self._check_dim_nl_tensor(p, q, mpq, 'M', self.dim['state'])
        for (p, q), npq in self.npq.items():
            if self.mode == 'function':
                self._check_dim_nl_fct(p, q, npq, 'N', self.dim['output'])
            else:
                self._check_dim_nl_tensor(p, q, npq, 'M', self.dim['output'])
        # If no error is raised, return True
        return True


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


    def _check_dim_nl_fct(self, p, q, fct, name, dim_result):
        """Verify shape and functionnality of the multilinear functions."""
        str_fct = '{}_{}{} function: '.format(name, p, q)
        # Check that each nonlinear lambda functions:
        # - accepts the good number of input arguments
        assert fct.__code__.co_argcount == p + q, \
               str_fct + 'wrong number of input arguments ' + \
               '(got {}, expected {}).'.format(fct.__code__.co_argcount, p + q)
        try:
            state_vectors = (np.ones(self.dim['state']),)*p
            input_vectors = (np.ones(self.dim['input']),)*q
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
        assert len(result_vector) == dim_result, \
               str_fct + 'wrong shape for the output (got ' + \
               '{}, expected {}).'.format(result_vector.shape, (dim_result,1))


    def _check_dim_nl_tensor(self, p, q, tensor, name, dim_result):
        """Verify shape and functionnality of the multilinear tensors."""
        str_tensor = '{}_{}{} tensor: '.format(name, p, q)
        shape = tensor.shape
        # Check that each nonlinear lambda functions:
        # - accepts the good number of input arguments
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


    def _is_linear(self):
        """Check if the system is linear."""
        return len(self.mpq) == 0 and len(self.npq) == 0

    #=============================================#

    def compute_volterra_kernels(self, fs, T, order_max=2, which='time'):
        #TODO sauvegarde des noyaux input2state
        #TODO faire marcher en mode 'function'
        #TODO tester pour dim['entree'/'sortie'] diff de 1
        #TODO faire docstring
        N = int(fs*T + 1)
        if which == 'time' or which == 'both':
            self._compute_time_kernels(fs, T, order_max)
        if which == 'freq':
            self._compute_frequency_kernels(fs, N, order_max)
        elif which == 'both':
            self._compute_frequency_kernels(fs, N, order_max, from_time=True)


    def _compute_time_kernels(self, fs, T, order_max):
        #TODO faire pour holder1
        #TODO faire marcher en mode 'function'
        #TODO prendre en compte les Npq
        #TODO faire docstring

        from scipy import linalg
        from pyvi.tools.combinatorics import make_dict_pq_set

        time_vec = np.arange(0, T + (1/fs), step=1/fs)
        N = time_vec.shape[0]

        # Initialization
        self._time_vector = time_vec
        self._time_in2state = dict()
        self.volterra_kernels = dict()
        for n in range(1, order_max+1):
            self._time_in2state[n] = np.zeros((self.dim['state'],)+(N,)*n)

        # Filter computation
        w_filter = np.zeros((N, self.dim['state'], self.dim['state']))
        for ind in range(N):
            w_filter[ind] = linalg.expm(self.A_m * time_vec[ind])
        w = w_filter[1]
        A_inv = np.linalg.inv(self.A_m)
        holder0_bias = A_inv.dot(w) - A_inv

        # Order 1
        self._time_in2state[1][:, 1:] = np.squeeze(np.dot(w_filter[:-1],
                                                          np.dot(holder0_bias,
                                                                 self.B_m)).T)

        # Compute list of Mpq combinations and tensors/functions
        dict_mpq_set = make_dict_pq_set(self.is_mpq_used, order_max,
                                        self.sym_bool)
        dirac4input = np.zeros((self.dim['input'], N))
        dirac4input[0] = 1

        # Computation of the Mpq/Npq functions (given as tensors)
        def pq_computation(n, p, q, order_set, dict_pq):
            temp_arg = ()
            min_ind = p + q
            for count in range(p):
                max_ind = min_ind + order_set[count]
                temp_arg += (self._time_in2state[order_set[count]],)
                temp_arg += ([count] + list(range(min_ind, max_ind)),)
                min_ind = max_ind
            for count in range(q):
                temp_arg += (dirac4input, [p+count, min_ind+count])
            temp_arg += (list(range(p + q + n)),)
            return np.tensordot(dict_pq[(p, q)], np.einsum(*temp_arg), p+q)

        # Correction of the bias due to ADC converter (with holder of order 0)
        def holder_bias(mpq_output, n):
            idx = [slice(None)] + [slice(N-1)] * n
            return np.tensordot(holder0_bias, mpq_output[idx], 1)

        # Loop (dynamical equation)
        for n, elt in dict_mpq_set.items():
            for p, q, order_set, nb in elt:
                mpq_output = nb * pq_computation(n, p, q, order_set, self.mpq)
                idx = [slice(None)] + [slice(1, N)] * n
                self._time_in2state[n][idx] += holder_bias(mpq_output, n)

        # Symmetrization
        for ind in range(self.dim['state']):
            self._time_in2state[n][ind] = symmetrization( \
                                                   self._time_in2state[n][ind])

        # Recursive term
        for n in range(2, order_max+1):
            for ind in range(n,n*(N-1)+1):
                for indexes in itr.filterfalse(lambda x: sum(x)-ind,
                                               itr.product(range(1, N),
                                                           repeat=n)):
                    idx_in = [slice(None)] + [(m-1) for m in indexes]
                    idx_out = [slice(None)] + list(indexes)
                    self._time_in2state[n][idx_out] += \
                            np.tensordot(w, self._time_in2state[n][idx_in], 1)

        # Output equation (only linear term C_m)
        for n in range(1, order_max+1):
            self.volterra_kernels[n] = np.squeeze( \
                                         np.tensordot(self.C_m,
                                                      self._time_in2state[n],
                                                      1))


    def _compute_frequency_kernels(self, fs, N, order_max, from_time=False):
        #TODO methode generale superieur a l'ordre 2
        #TODO faire marcher en mode 'function'
        #TODO prendre en compte les Npq
        #TODO faire docstring

        # Initialization
        freq_vec = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))

        self._frequency_vector = freq_vec
        self._freq_in2state = dict()
        self.transfer_kernels = dict()

        if from_time:
            for n in range(1, order_max+1):
                self._freq_in2state[n] = np.fft.fftshift( \
                                     np.fft.fftn(self._time_in2state[n],
                                                 axes=np.arange(1, n+1)))
                self.transfer_kernels[n] = np.fft.fftshift( \
                                     np.fft.fftn(self.volterra_kernels[n]))

        else:
            # Filter computation
            def _filter_values(f):
                fac = np.reshape(2j*np.pi*f, f.shape + (1, 1))
                identity = np.identity(self.dim['state'])
                return np.linalg.inv(fac * identity - self.A_m)

            w = _filter_values(freq_vec)
            holder0_bias = np.exp(- 1j * np.pi / fs) * np.sinc(freq_vec/fs)

            # Order 1
            self._freq_in2state[1] = np.squeeze(np.dot(holder0_bias * w,
                                                       self.B_m)).T
            self.transfer_kernels[1] = np.squeeze( \
                                        np.dot(self.C_m,
                                               self._freq_in2state[1]) + \
                                        self.D_m)

            # Order 2
            ones4input = holder0_bias
            temp_state = np.zeros((self.dim['state'], N, N), dtype='complex128')
            if self.is_mpq_used(2, 0):
                temp_tensor = np.einsum(self._freq_in2state[1], (0, 2),
                                        self._freq_in2state[1], (1, 3),
                                        (0, 1, 2, 3))
                temp_state += np.tensordot(self.mpq[(2, 0)], temp_tensor, 2)
            if self.is_mpq_used(1, 1):
                temp_tensor = np.einsum(self._freq_in2state[1], (0, 2),
                                        ones4input, (1, 3), (0, 1, 2, 3))
                temp_result = np.tensordot(self.mpq[(1, 1)], temp_tensor, 2)
                temp_result += np.swapaxes(temp_result, 1, 2)
                temp_state += (1/2) * temp_result
            if self.is_mpq_used(0, 2):
                temp_tensor = np.einsum(ones4input, (0, 2),
                                        ones4input, (1, 3), (0, 1, 2, 3))
                temp_state += np.tensordot(self.mpq[(0, 2)], temp_tensor, 2)
            freq_somme = freq_vec[:, np.newaxis] + freq_vec[np.newaxis, :]
            self._freq_in2state[2] = np.einsum('ijkl,kij->lij',
                                               _filter_values(freq_somme),
                                               temp_state)
            self.transfer_kernels[2] = np.squeeze( \
                                        np.tensordot(self.C_m,
                                                     self._freq_in2state[2], 1))


    def plot_kernels(self):
        #TODO faire plots pour noyau temporel
        #TODO plot + beau (grilles, axes, titres, labels, ticks, ...)
        #TODO faire save
        #TODO faire docstring
        #TODO faire appel à fct dans plotbox

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        if 'volterra_kernels' in self.__dict__:

            # Order 1
            plt.figure('Volterra kernel of order 1 (linear filter)')
            plt.clf()
            plt.plot(self._time_vector, self.volterra_kernels[1])

            # Order 2
            time_x, time_y = np.meshgrid(self._time_vector, self._time_vector)

            plt.figure('Volterra kernel of order 2 (a)')
            plt.clf()
            N = 20
            plt.contourf(time_x, time_y, self.volterra_kernels[2], N)
            plt.colorbar(extend='both')

            plt.figure('Volterra kernel of order 2 (b)')
            plt.clf()
            ax = plt.subplot(111, projection='3d')
            surf = ax.plot_surface(time_x, time_y, self.volterra_kernels[2],
                                   linewidth=0.1, antialiased=True, cmap='jet',
                                   rstride=1, cstride=1)
            plt.colorbar(surf, extend='both')

            plt.figure('Volterra kernel of order 2 (c)')
            plt.clf()
            ax = plt.subplot(111, projection='3d')
            ax.plot_wireframe(time_x, time_y, self.volterra_kernels[2],
                              linewidth=0.1, antialiased=True, cmap='jet')

        if 'transfer_kernels' in self.__dict__:

            # Order 1
            H1_amp_db = 20*np.log10(np.abs(self.transfer_kernels[1]))
            H1_phase = np.angle(self.transfer_kernels[1])

            plt.figure('Transfer kernel of order 1 (linear filter)')
            plt.clf()
            plt.subplot(211)
            plt.semilogx(self._frequency_vector, H1_amp_db, basex=10)
            plt.title('Magnitude')
            plt.subplot(212)
            plt.semilogx(self._frequency_vector, H1_phase, basex=10)
            plt.title('Phase')

            # Order 2
            H2_amp_db = 20*np.log10(np.abs(self.transfer_kernels[2]))
            H2_phase = np.angle(self.transfer_kernels[2])
            freq_x, freq_y = np.meshgrid(self._frequency_vector,
                                         self._frequency_vector)

            plt.figure('Transfer kernel of order 2 (a)')
            plt.clf()
            N = 20
            plt.subplot(211)
            plt.contourf(freq_x, freq_y, H2_amp_db, N)
            plt.colorbar(extend='both')
            plt.title('Magnitude (dB)')
            plt.subplot(212)
            plt.contourf(freq_x, freq_y, H2_phase, N)
            plt.colorbar(extend='both')
            plt.title('Phase')

            plt.figure('Transfer kernel of order 2 (b)')
            plt.clf()
            ax = plt.subplot(211, projection='3d')
            surf = ax.plot_surface(freq_x, freq_y, H2_amp_db,
                                   linewidth=0.1, antialiased=True, cmap='jet',
                                   rstride=1, cstride=1)
            plt.colorbar(surf, extend='both')
            plt.title('Magnitude (dB)')
            ax = plt.subplot(212, projection='3d')
            surf = ax.plot_surface(freq_x, freq_y, H2_phase,
                                   linewidth=0.1, antialiased=True, cmap='jet',
                                   rstride=1, cstride=1)
            plt.colorbar(surf, extend='both')
            plt.title('Phase')

            plt.figure('Transfer kernel of order 2 (c)')
            plt.clf()
            ax = plt.subplot(211, projection='3d')
            ax.plot_wireframe(freq_x, freq_y, H2_amp_db,
                              linewidth=0.1, antialiased=True, cmap='jet')
            plt.title('Magnitude (dB)')
            ax = plt.subplot(212, projection='3d')
            ax.plot_wireframe(freq_x, freq_y, H2_phase,
                              linewidth=0.1, antialiased=True, cmap='jet')
            plt.title('Phase')

        plt.show()


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

        # Check dimension and linearity
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


#==============================================================================
# Functions
#==============================================================================

def symmetrization(array):
    """
    Symmetrize a multidimensional square array (each dimension must have the
    same length).

    Parameters
    ----------
    array : numpy.ndarray
        Array to symmetrize.

    Returns
    -------
    array_sym : numpy.ndarray
        Symmetrized array.
    """

    from math import factorial

    shape = array.shape
    assert len(set(shape)) == 1, 'Multidimensional array is not square ' + \
        '(has shape {})'.format(shape)
    n = len(array.shape)

    array_sym = np.zeros(shape, dtype=array.dtype)
    for ind in itr.permutations(range(n), n):
        array_sym += np.transpose(array, ind)
    return array_sym / factorial(n)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    import pyvi.simulation.systems as systems
    from pyvi.tools.plotbox import plot_kernel_time

    print(systems.system_test(mode='tensor'))
    print(systems.loudspeaker_sica(mode='function'))

    system = systems.second_order_w_nl_damping(gain=1, f0=100, damping=0.2,
                                               nl_coeff=[1e-1, 3e-5])
    fs = 1000
    T = 0.015
    system.compute_volterra_kernels(fs, T, order_max=3, which='both')
#    system.plot_kernels()

    plot_kernel_time(system._time_vector, system.volterra_kernels[1])
    plot_kernel_time(system._time_vector, system.volterra_kernels[2],
                     style='wireframe')

