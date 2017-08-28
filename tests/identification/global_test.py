# -*- coding: utf-8 -*-
"""
Global test script for pyvi/identification module.

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

from mytoolbox.utilities.misc import (my_parse_arg_for_tests,
                                      submodule_test_calls)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    indent = my_parse_arg_for_tests()
    submodule_test_calls(['tools', 'methods'], __file__, indent)
