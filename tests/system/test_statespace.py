# -*- coding: utf-8 -*-
"""
Test script for pyvi/system/statespace.py

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 27 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import warnings
import numpy as np
from pyvi.system.statespace import (StateSpace, NumericalStateSpace,
                                    SymbolicStateSpace)
import pyvi.system.dict as systems
from mytoolbox.utilities.misc import my_parse_arg_for_tests


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    indent = my_parse_arg_for_tests()


    #######################
    ## Matrix dimensions ##
    #######################

    print(indent + 'Testing StateSpace class...', end=' ')
    def dim_check(text, shapeA, shapeB, shapeC, shapeD,
                  mpq=dict(), npq=dict()):
        try:
            StateSpace(np.empty(shapeA), np.empty(shapeB),
                       np.empty(shapeC), np.empty(shapeD),
                       mpq=mpq, npq=npq)
            assert text
        except AssertionError:
            pass

    message = 'Error in checking consistency of {} dimension.'
    dim_check('Error in checking A is square.',
              (3, 3, 3), (3, 2), (2, 3), (2, 2))
    dim_check('Error in checking A is square.', (3, 4), (3, 2), (2, 3), (2, 2))
    dim_check(message.format('state'),
              (3, 3), (4, 2), (2, 3), (2, 2))
    dim_check(message.format('state'), (3, 3), (4), (2, 3), (2, 1))
    dim_check(message.format('state'), (3, 3), (3, 2), (2, 4), (2, 2))
    dim_check(message.format('state'), (3, 3), (3, 2), (4), (1, 2))
    dim_check(message.format('input'), (3, 3), (3, 2), (2, 3), (2, 4))
    dim_check(message.format('input'), (3, 3), (3,), (2, 3), (2, 2))
    dim_check(message.format('input'), (3, 3), (3, 2), (2, 3), (2,))
    dim_check(message.format('output'), (3, 3), (3, 2), (2, 3), (4, 2))
    dim_check(message.format('output'), (3, 3), (3,), (2, 3), (2, 2))
    dim_check(message.format('output'), (3, 3), (3, 2), (2, 3), (2))


    ###############################
    ## Tensor dimension checking ##
    ###############################

    message = 'Error in checking dimensions of Mpq tensors.'
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              mpq={(2, 0): np.empty((3, 3, 4))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              mpq={(2, 0): np.empty((4, 3, 3))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              mpq={(1, 1): np.empty((3, 3, 1))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              mpq={(1, 1): np.empty((3, 4, 2))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              mpq={(1, 1): np.empty((4, 3, 2))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              mpq={(0, 2): np.empty((3, 2, 1))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              mpq={(0, 2): np.empty((4, 2, 2))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              npq={(2, 0): np.empty((2, 3, 4))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              npq={(2, 0): np.empty((3, 3, 3))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              npq={(1, 1): np.empty((2, 3, 1))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              npq={(1, 1): np.empty((2, 4, 2))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              npq={(1, 1): np.empty((3, 3, 2))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              npq={(0, 2): np.empty((2, 2, 1))})
    dim_check(message, (3, 3), (3, 2), (2, 3), (2, 2),
              npq={(0, 2): np.empty((3, 2, 2))})


    ###################################
    ## Warnings for non-SISO systems ##
    ###################################

    warnings.filterwarnings("error")

    def check_warnings(n_in, n_s, n_out):
        try:
            StateSpace(np.empty((n_s, n_s)), np.empty((n_s, n_in)),
                       np.empty((n_out, n_s)), np.empty((n_out, n_in)))
            assert 'Warning not raised about non-SISO systems.'
        except UserWarning:
            pass

    check_warnings(3, 2, 1)
    check_warnings(3, 1, 2)
    check_warnings(3, 2, 2)


    #######################
    ## Attribute '_type' ##
    #######################

    warnings.filterwarnings("ignore")

    def check_type(sys_type, n_in, n_s, n_out):
        tmp = StateSpace(np.empty((n_s, n_s)), np.empty((n_s, n_in)),
                         np.empty((n_out, n_s)), np.empty((n_out, n_in)))
        assert tmp._type == sys_type, \
            "Error in computation of attribute '_type'."

    check_type('SISO', 1, 3, 1)
    check_type('MISO', 2, 3, 1)
    check_type('SIMO', 1, 3, 2)
    check_type('MIMO', 2, 3, 2)


    ####################
    ## Categorization ##
    ####################

    warnings.filterwarnings("default")

    def check_categories(expected_values, mpq_present=[], npq_present=[]):
        mpq = dict()
        for p, q in mpq_present:
            mpq[(p, q)] = np.empty((3,) + (3,)*p + (1,)*q)
        npq = dict()
        for p, q in npq_present:
            npq[(p, q)] = np.empty((1,) + (3,)*p + (1,)*q)
        tmp = StateSpace(np.empty((3, 3)), np.empty((3, 1)),
                         np.empty((1, 3)), np.empty((1, 1)),
                         mpq=mpq, npq=npq)
        assert tmp.linear == expected_values[0], \
            "Error in computation of attribute 'linear'."
        assert tmp.state_eqn_linear_analytic == expected_values[1], \
            "Error in computation of attribute 'state_eqn_linear_analytic'."
        assert tmp.dynamical_nl_only_on_state == expected_values[2], \
            "Error in computation of attribute 'dynamical_nl_only_on_state'."

    check_categories([True, True, True], [], [])
    check_categories([False, True, True], [(2, 0)], [])
    check_categories([False, True, True], [], [(2, 0)])
    check_categories([False, True, True], [(2, 0)], [(2, 0)])
    check_categories([False, True, True], [(2, 0)], [(1, 1)])
    check_categories([False, True, True], [(2, 0)], [(0, 2)])
    check_categories([False, True, False], [(1, 1)], [])
    check_categories([False, True, False], [(1, 1)], [(2, 0)])
    check_categories([False, True, False], [(1, 1)], [(1, 1)])
    check_categories([False, True, False], [(1, 1)], [(0, 2)])
    check_categories([False, False, False], [(0, 2)], [])
    check_categories([False, False, False], [(0, 2)], [(2, 0)])
    check_categories([False, False, False], [(0, 2)], [(1, 1)])
    check_categories([False, False, False], [(0, 2)], [(0, 2)])
    print('Done')


    #############################
    ## Attribute 'nl_colinear' ##
    #############################

    print(indent + 'Testing NumericalStateSpace class...', end=' ')
    A = np.empty((3, 3))
    B = np.zeros((3, 1))
    C = np.empty((1, 3))
    D = np.empty((1, 1))
    m20 = np.zeros((3, 3, 3))
    B[1] = 1
    m20[1, 1, 2] = 10
    m20[1, 2, 1] = 10
    m20[1, 1, 1] = 5
    tmp_t1 = NumericalStateSpace(A, B, C, D, mpq={(2, 0): m20})
    m20[1] = 2
    tmp_t2 = NumericalStateSpace(A, B, C, D, mpq={(2, 0): m20})
    m20[2, 1, 1] = 5
    tmp_f1 = NumericalStateSpace(A, B, C, D, mpq={(2, 0): m20})
    m20 = np.zeros((3, 3, 3))
    m20[0] = 2
    tmp_f2 = NumericalStateSpace(A, B, C, D, mpq={(2, 0): m20})
    message = "Error in computation of attribute 'nl_colinear'."
    assert tmp_t1.nl_colinear == True, message
    assert tmp_t2.nl_colinear == True, message
    assert tmp_f1.nl_colinear == False, message
    assert tmp_f2.nl_colinear == False, message

    sys_num = systems.create_test(mode='numeric')
    print('Done')


    ################################################################
    ## Equality between Numerical and Symbolic StateSpace objects ##
    ################################################################

    print(indent + 'Testing SymbolicalStateSpace class...', end=' ')
    sys_symb = systems.create_test(mode='symbolic')
    message = "Similar Numerical and Symbolic StateSpace objects have " + \
              "different attribute '{}'."
    assert sys_num._type == sys_symb._type, message.format('_type')
    assert sys_num._single_input == sys_symb._single_input, \
        message.format('_single_input')
    assert sys_num._single_output == sys_symb._single_output, \
        message.format('_single_output')
    assert sys_num._dim_ok == sys_symb._dim_ok, message.format('_dim_ok')
    assert sys_num._state_eqn_linear == sys_symb._state_eqn_linear, \
        message.format('_state_eqn_linear')
    assert sys_num._output_eqn_linear == sys_symb._output_eqn_linear, \
        message.format('_output_eqn_linear')
    assert sys_num.linear == sys_symb.linear, message.format('linear')
    assert sys_num.state_eqn_linear_analytic == \
        sys_symb.state_eqn_linear_analytic, \
        message.format('state_eqn_linear_analytic')
    assert sys_num.dynamical_nl_only_on_state == \
        sys_symb.dynamical_nl_only_on_state, \
        message.format('dynamical_nl_only_on_state')
    print('Done')

