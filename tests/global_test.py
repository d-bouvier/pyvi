# -*- coding: utf-8 -*-
"""
Global test script for pyvi package.

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

import os


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    default_path = os.getcwd()
    test_base_path = os.path.abspath(os.path.dirname(__file__))
    list_modules = ['utilities', 'system', 'simulation', 'separation',
                    'identification']
    print('Package: pyvi')
    for module in list_modules:
        print('   Module:', module)
        os.chdir(test_base_path + os.sep + module)
        os.system('python global_test.py --indentation 6')
    os.chdir(default_path)