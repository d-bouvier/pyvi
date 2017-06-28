# -*- coding: utf-8 -*-
"""
Test script for and pyvi.system.dict

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

import pyvi.system.dict as systems


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """


    sys_num = systems.create_test(mode='numeric')
    print(sys_num)
    sys_symb = systems.create_test(mode='symbolic')
    print(sys_symb)

    print(systems.create_loudspeaker_sica())
    print(systems.create_nl_damping())