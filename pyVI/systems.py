# -*- coding: utf-8 -*-
"""
Module that gives functions defining physical or theoretical systems.

Functions for system parameters
-------------------------------
loudspeaker_sica :
    Returns a StateSpace object corresponding to the SICA Z000900 loudspeaker.
simple_system :
    Returns a StateSpace object corresponding to a simple system for simulation
    test.

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

from pyvi.system.statespace import StateSpace
import numpy as np

#==============================================================================
# System parameters
#==============================================================================

def loudspeaker_sica(version='tristan', output='pos', mode='function'):
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
