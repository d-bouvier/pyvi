# -*- coding: utf-8 -*-
"""
Global test script for pyvi/utilities module.

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
import argparse


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


    #######################
    ## Call test scripts ##
    #######################

    default_path = os.getcwd()
    test_base_path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(test_base_path)
    command = 'python test_{}.py --indentation ' + str(3 + indent)
    list_files = ['mathbox', 'plotbox', 'savebox']
    for file_name in list_files:
        print(ss + 'File:', file_name + '.py')
        os.system(command.format(file_name))
    os.chdir(default_path)
