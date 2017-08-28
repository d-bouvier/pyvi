# -*- coding: utf-8 -*-
"""
Global test script for pyvi package.

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

from mytoolbox.utilities.misc import package_test_calls


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    package_test_calls(['utilities', 'system', 'simulation', 'separation',
                        'identification'], __file__)