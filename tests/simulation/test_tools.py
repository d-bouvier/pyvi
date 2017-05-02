# -*- coding: utf-8 -*-
"""
Test script for and pyvi.simulation.tools

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 2 May. 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

from pyvi.system.dict import test


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    test_sys = test()
    test_sys._create_simulation_parameters(fs=2000, holder_order=1)
    print(test_sys._simu.__dict__)
