# -*- coding: utf-8 -*-
"""
Test script for pyvi/system/dict.py

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 21 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import argparse
import pyvi.system.dict as systems


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    #####################
    ## Parsing options ##
    #####################

    parser = argparse.ArgumentParser()
    parser.add_argument('-ind', '--indentation', type=int, default=0)
    args = parser.parse_args()
    indent = args.indentation
    ss = ' ' * indent


    ########################
    ## Systems dictionary ##
    ########################

    print(ss + 'Testing create_loudspeaker_sica()...', end=' ')
    loudspeaker_sica = systems.create_loudspeaker_sica()
    message = 'Error in create_loudspeaker_sica().'
    assert loudspeaker_sica.dim['input'] == 1, message
    assert loudspeaker_sica.dim['state'] == 3, message
    assert loudspeaker_sica.dim['output'] == 1, message
    assert loudspeaker_sica._type == 'SISO', message
    assert not loudspeaker_sica.linear, message
    assert loudspeaker_sica._output_eqn_linear, message
    assert loudspeaker_sica.state_eqn_linear_analytic, message
    assert loudspeaker_sica.dynamical_nl_only_on_state, message
    assert not loudspeaker_sica.nl_colinear, message
    print('Done')


    print(ss + 'Testing create_moog()...', end=' ')
    moog_1 = systems.create_moog(taylor_series_truncation=1)
    moog_2 = systems.create_moog(taylor_series_truncation=2)
    moog_3 = systems.create_moog(taylor_series_truncation=3)
    tmp_3 = {(3, 0): 0, (2, 1): 0, (1, 2): 0, (0, 3): 0}
    moog_5 = systems.create_moog(taylor_series_truncation=5)
    tmp_5 = {(5, 0): 0, (4, 1): 0, (3, 2): 0, (2, 3): 0, (1, 4): 0, (0, 5): 0,
             (3, 0): 0, (2, 1): 0, (1, 2): 0, (0, 3): 0}
    message = 'Error in create_moog().'
    assert moog_1.dim['input'] == 1, message
    assert moog_1.dim['state'] == 4, message
    assert moog_1.dim['output'] == 1, message
    assert moog_1._type == 'SISO', message
    assert moog_1.linear, message
    for key, val_1 in moog_1.__dict__.items():
        if (key in ['A_m', 'B_m', 'C_m', 'D_m']) and (key in moog_2.__dict__):
            assert (val_1 == moog_2.__dict__[key]).all(), message
        elif key in moog_2.__dict__:
            assert val_1 == moog_2.__dict__[key], message
        else:
            assert False, message
    assert not moog_3.linear, message
    assert moog_3._output_eqn_linear, message
    assert not moog_3.state_eqn_linear_analytic, message
    assert not moog_3.dynamical_nl_only_on_state, message
    assert not moog_3.nl_colinear, message
    assert moog_3.mpq.keys() == tmp_3.keys(), message
    assert not moog_5.linear, message
    assert moog_5._output_eqn_linear, message
    assert not moog_5.state_eqn_linear_analytic, message
    assert not moog_5.dynamical_nl_only_on_state, message
    assert not moog_5.nl_colinear, message
    assert moog_5.mpq.keys() == tmp_5.keys(), message
    print('Done')


    print(ss + 'Testing create_nl_damping()...', end=' ')
    second_order = systems.create_nl_damping()
    message = 'Error in create_nl_damping().'
    assert second_order.dim['input'] == 1, message
    assert second_order.dim['state'] == 2, message
    assert second_order.dim['output'] == 1, message
    assert second_order._type == 'SISO', message
    assert not second_order.linear, message
    assert second_order._output_eqn_linear, message
    assert second_order.state_eqn_linear_analytic, message
    assert second_order.dynamical_nl_only_on_state, message
    assert second_order.nl_colinear, message
    print('Done')


    print(ss + 'Testing create_test()...', end=' ')
    sys_num = systems.create_test(mode='numeric')
    sys_symb = systems.create_test(mode='symbolic')
    message = 'Error in create_test().'
    assert not sys_num.linear, message
    assert not sys_symb.linear, message
    print('Done')
