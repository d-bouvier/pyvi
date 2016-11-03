# -*- coding: utf-8 -*-
"""
Paths for the pyVI package.

Notes
-----
@author:    bouvier@ircam.fr
            Damien Bouvier, IRCAM, Paris

Last modified on 3 Nov. 2016
Developed for Python 3.5.1

"""

#==============================================================================
#Importations
#==============================================================================

import os
import datetime
import pickle
from numpy import savez

#==============================================================================
# Global variables
#==============================================================================

__data_directory__ = os.path.abspath(os.path.join(os.path.dirname( \
                        os.path.dirname(os.path.realpath(__file__))), 'data'))


#==============================================================================
# Functions
#==============================================================================

def folder_check(folder_abs_path):
    """
    Check if the given path indicates a folder, and creates it if not.
    """

    if not os.path.isdir(folder_abs_path):
        os.mkdir(folder_abs_path)

def name_save_files(folder, name):
    """
    Returns two string with date and label for data saving.
    """

    folder_check(__data_directory__,)
    folder_check(os.path.abspath(os.path.join(__data_directory__, folder)))

    date_str = datetime.datetime.now().strftime('%Y_%m_%d_%Hh%M')
    common_name = name + '_' + date_str
    name1 = common_name + '.config'
    name2 = common_name + '.npz'
    str1 = os.path.abspath(os.path.join(__data_directory__, folder, name1))
    str2 = os.path.abspath(os.path.join(__data_directory__, folder, name2))
    return str1, str2

def save_data(folder, name, param_dict, array_dict):
    """
    Save parameters data and numpy arrays into 2 different files.
    """

    str1, str2 = name_save_files(folder, name)

    pickle.dump(param_dict, open(str1, 'wb'))
    savez(str2, **array_dict)
