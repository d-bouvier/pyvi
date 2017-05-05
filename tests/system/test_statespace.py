# -*- coding: utf-8 -*-
"""
Test script for and pyvi.system.statespace

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 04 May 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import pyvi.system.dict as systems


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    sys_num = systems.test(mode='numeric')
    sys_symb = systems.test(mode='symbolic')
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