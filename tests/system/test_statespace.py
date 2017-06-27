# -*- coding: utf-8 -*-
"""
Test script for and pyvi.system.statespace

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 27 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from pyvi.system.statespace import StateSpace, NumericalStateSpace
import pyvi.system.dict as systems


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    sys_num = systems.create_test(mode='numeric')
    sys_symb = systems.create_test(mode='symbolic')
    assert sys_num._type == sys_symb._type, \
        "Similar Numerical and Symbolic StateSpace objects have different " + \
        "attribute '_type'"
    assert sys_num._single_input == sys_symb._single_input, \
        "Similar Numerical and Symbolic StateSpace objects have different " + \
        "attribute '_single_input'"
    assert sys_num._single_output == sys_symb._single_output, \
        "Similar Numerical and Symbolic StateSpace objects have different " + \
        "attribute '_single_output'"
    assert sys_num._dim_ok == sys_symb._dim_ok, \
        "Similar Numerical and Symbolic StateSpace objects have different " + \
        "attribute '_dim_ok'"

    N = 3
    for dim_in in [1, 2]:
        for dim_out in [1, 2]:
            test_system = StateSpace(np.zeros((N, N)), np.zeros((N, dim_in)),
                                     np.zeros((dim_out, N)),
                                     np.zeros((dim_out, dim_in)))
            print('Dimensions (I/S/O): {}, {}, {})'.format(dim_in, N, dim_out))
            print('              dim :', test_system.dim)
            print('            _type :', test_system._type)
            print('    _single_input :', test_system._single_input)
            print('   _single_output :', test_system._single_output)


    d_in = 1
    d_out = 1
    A = np.zeros((N, N))
    B = np.zeros((N, d_in))
    C = np.zeros((d_out, N))
    D = np.zeros((d_out, d_in))
    m20 = np.zeros((N, N, N))
    m11 = np.zeros((N, N, d_in))
    m02 = np.zeros((N, d_in, d_in))
    n20 = np.zeros((d_out, N, N))

    def f_print(system: StateSpace):
        print('Mpq                             :', system.mpq.keys())
        print('Npq                             :', system.npq.keys())
        print('_is_linear                      :', system.linear)
        print('_is_state_eqn_linear_analytic   :',
              system.state_eqn_linear_analytic)
        print('_are_dynamical_nl_only_on_state :',
              system.dynamical_nl_only_on_state)
        print('_are_nl_colinear                :', system.nl_colinear)
        print()
    # Linear system
    f_print(NumericalStateSpace(A, B, C, D))
    # State equation linear-analytic & nonlinearities only on state
    f_print(NumericalStateSpace(A, B, C, D, npq_dict={(2, 0): n20}))
    f_print(NumericalStateSpace(A, B, C, D, mpq_dict={(2, 0): m20}))
    # State equation linear-analytic
    f_print(NumericalStateSpace(A, B, C, D, mpq_dict={(1, 1): m11}))
    f_print(NumericalStateSpace(A, B, C, D,
                                mpq_dict={(2, 0): m20, (1, 1): m11}))
    # State equation not linear-analytic
    f_print(NumericalStateSpace(A, B, C, D, mpq_dict={(0, 2): m02}))
    f_print(NumericalStateSpace(A, B, C, D,
                                mpq_dict={(2, 0): m20, (0, 2): m02}))
    # "Colinear" system
    B[1] = 1
    m20[1, 1, 2] = 10
    m20[1, 2, 1] = 10
    m20[1, 1, 1] = 5
    f_print(NumericalStateSpace(A, B, C, D, mpq_dict={(2, 0): m20}))
    m20[1] = 2
    f_print(NumericalStateSpace(A, B, C, D, mpq_dict={(2, 0): m20}))
    # "Non-colinear" system
    m20[2, 1, 1] = 5
    f_print(NumericalStateSpace(A, B, C, D, mpq_dict={(2, 0): m20}))
    m20 = np.zeros((N, N, N))
    m20[0] = 2
    f_print(NumericalStateSpace(A, B, C, D, mpq_dict={(2, 0): m20}))