# -*- coding: utf-8 -*-
"""
Test script for pyvi.system.dict

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

    print(ss + 'Testing create_test()...', end=' ')
    sys_num = systems.create_test(mode='numeric')
    sys_symb = systems.create_test(mode='symbolic')
    print('Done')

    print(ss + 'Testing create_loudspeaker_sica()...', end=' ')
    loudspeaker_sica = systems.create_loudspeaker_sica()
    message = 'Error in create_loudspeaker_sica().'
    assert loudspeaker_sica.dim['input'] == 1, message
    assert loudspeaker_sica.dim['state'] == 3, message
    assert loudspeaker_sica.dim['output'] == 1, message
    assert loudspeaker_sica._type == 'SISO', message
    assert loudspeaker_sica._output_eqn_linear, message
    assert loudspeaker_sica.state_eqn_linear_analytic, message
    assert loudspeaker_sica.dynamical_nl_only_on_state, message
    assert not loudspeaker_sica.nl_colinear, message
    print('Done')

    print(ss + 'Testing create_nl_damping()...', end=' ')
    second_order = systems.create_nl_damping()
    message = 'Error in create_nl_damping().'
    assert second_order.dim['input'] == 1, message
    assert second_order.dim['state'] == 2, message
    assert second_order.dim['output'] == 1, message
    assert second_order._type == 'SISO', message
    assert second_order._output_eqn_linear, message
    assert second_order.state_eqn_linear_analytic, message
    assert second_order.dynamical_nl_only_on_state, message
    assert second_order.nl_colinear, message
    print('Done')