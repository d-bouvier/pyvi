# -*- coding: utf-8 -*-
"""
Module that gives functions creating physical or theoretical systems.

Functions for system parameters
-------------------------------
create_loudspeaker_sica :
    Returns NumericalStateSpace object corresponding to the SICA Z000900
    loudspeaker.
create_nl_damping :
    Returns a NumericalStateSpace object corresponding to a second_order system
    with nonlinear stiffness.
create_test :
    Returns either a NumricalStateSpace or SymbolicStateSpace object of a
    theoretical system for tests.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 25 Apr. 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

from .statespace import NumericalStateSpace, SymbolicStateSpace
import numpy as np
from sympy import symbols, Matrix
from sympy.tensor.array import MutableDenseNDimArray


#==============================================================================
# System parameters
#==============================================================================

def create_loudspeaker_sica(version='tristan', output='pos'):
    """
    Function that create and returns the StateSpace object corresponding to the
    SICA Z000900 loudspeaker
    (http://www.sica.it/media/Z000900C.pdf551d31b7b491e.pdf).

    Parameters
    ----------
    version : {'tristan', 'CFA'}, optional
        Version to simulate.
    output : {'pos', 'current'}, optional
        Defines the output of the system

    Returns
    -------
    Object of class NumericalStateSpace.

    """

    ## Physical parameters ##
    # Electric parameters
    if version == 'tristan': # Electodynamic driving parameter [T.m]
        Bl = 2.9
    elif version == 'CFA':
        Bl = 2.99
    Re = 5.7 # Electrical resistance of voice coil   [Ohm]
    Le = 0.11e-3 # Coil inductance [H]
    # Mechanical parameters
    Mms = 1.9e-3 # Mechanical mass [kg]
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

     # Dictionnaries of Mpq & Npq tensors
    m20 = np.zeros((3, 3, 3))
    m20[2, 1, 1] = -k[1]/Mms
    m30 = np.zeros((3, 3, 3, 3))
    m30[2, 1, 1, 1] = -k[2]/Mms

    mpq_dict = {(2, 0): m20, (3, 0): m30}
    npq_dict = dict()

    return NumericalStateSpace(A_m, B_m, C_m, D_m, mpq_dict, npq_dict,
                               pq_symmetry=True)


def create_nl_damping(gain=1, f0=100, damping=0.2, nl_coeff=[0, 1e-6]):
    """
    Function that create and returns the StateSpace object corresponding to a
    second order system with nonlinear stiffness.

    Parameters
    ----------
    gain : float, optional
        Gain at null frequency
    f0 : float, optional
        Natural frequency of the system
    damping : float, optional
        Damping factor of the system
    nl_coeff : [float, float, float], optional
        Coefficient associated respectively with M20, M30 and M40 functions

    Returns
    -------
    Object of class NumericalStateSpace.

    """

    w0 = 2 * np.pi * f0 # Natural pulsation

    # State-space matrices
    A_m = np.array([[0, 1],
                    [- w0**2, - 2 * damping * w0]]) # State-to-state matrix
    B_m = np.array([[0], [gain * w0**2]]); # Input-to-state matrix
    C_m = np.array([[1, 0]]) # State-to-output matrix
    D_m = np.zeros((1, 1)) # Input-to-output matrix

    # Dictionnaries of Mpq & Npq
    mpq_dict = dict()
    npq_dict = dict()

    p_count = 1
    for val in nl_coeff:
        p_count += 1
        temp_mpq = np.zeros((2,)*(p_count+1))
        temp_mpq[(1,)*(p_count+1)] = val
        mpq_dict[(p_count, 0)] = temp_mpq.copy()

    return NumericalStateSpace(A_m, B_m, C_m, D_m, mpq_dict, npq_dict,
                               pq_symmetry=True,)


def create_test(mode='numeric'):
    """
    Function that create and returns the StateSpace object corresponding to a
    simple system for testing and debugging the simulation.

    Returns
    -------
    Object of class NumericalStateSpace (if ``mode`` is 'numeric') or
    SymbolicStateSpace (if ``mode`` is 'symbolic').

    """

    if mode in ['numeric', 'num']:
        # State-space matrices
        A_m = np.array([[0, 1],
                        [- 10000, - 40]]) # State-to-state matrix
        B_m = np.array([[0], [1]]); # Input-to-state matrix
        C_m = np.array([[1, 0]]) # State-to-output matrix
        D_m = np.array([[1]]) # Input-to-output matrix

        # Mpq & Npq in 'tensor' mode
        m20 = np.zeros((2, 2, 2))
        m20[1, 1, 1] = 1
        m11 = np.zeros((2, 2, 1))
        m11[0, 1, 0] = -1
        m02 = np.zeros((2, 1, 1))
        m02[0, 0, 0] = 0.1

        n20 = np.zeros((1, 2, 2))
        n20[0, 1, 1] = 1
        n11 = np.zeros((1, 2, 1))
        n11[0, 1, 0] = -1
        n02 = np.zeros((1, 1, 1))
        n02[0, 0, 0] = -1

    elif mode in ['symbolic', 'symb']:
        a, b, c, d, e, f = symbols('a,b,c,d,e,f')
        ma, mb, mc, na, nb, nc = symbols('ma,mb,mc,na,nb,nc')

        # State-space matrices
        A_m = Matrix([[0, a],
                      [b, c]]) # State-to-state matrix
        B_m = Matrix([[0], [d]]); # Input-to-state matrix
        C_m = Matrix([[e, 0]]) # State-to-output matrix
        D_m = Matrix([[f]]) # Input-to-output matrix

        # Mpq & Npq in 'tensor' mode
        m20 = MutableDenseNDimArray(np.zeros(8), (2, 2, 2))
        m20[1, 1, 1] = ma
        m11 = MutableDenseNDimArray(np.zeros(4), (2, 2, 1))
        m11[0, 1, 0] = mb
        m02 = MutableDenseNDimArray(np.zeros(2), (2, 1, 1))
        m02[0, 0, 0] = mc

        n20 = MutableDenseNDimArray(np.zeros(4), (1, 2, 2))
        n20[0, 1, 1] = na
        n11 = MutableDenseNDimArray(np.zeros(2), (1, 2, 1))
        n11[0, 1, 0] = nb
        n02 = MutableDenseNDimArray(np.zeros(1), (1, 1, 1))
        n02[0, 0, 0] = nc

    # Dictionnaries of Mpq & Npq
    mpq_dict = {(2, 0): m20, (1, 1): m11, (0, 2): m02}
    npq_dict = {(2, 0): n20, (1, 1): n11, (0, 2): n02}


    if mode in ['numeric', 'num']:
        return NumericalStateSpace(A_m, B_m, C_m, D_m, mpq_dict, npq_dict,
                                   pq_symmetry=True)
    elif mode in ['symbolic', 'symb']:
        return SymbolicStateSpace(A_m, B_m, C_m, D_m, mpq_dict, npq_dict,
                                   pq_symmetry=True)