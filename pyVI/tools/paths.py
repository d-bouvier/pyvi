# -*- coding: utf-8 -*-
"""
Paths for the pyVI package.

Notes
-----
@author:    bouvier@ircam.fr
            Damien Bouvier, IRCAM, Paris

Last modified on 3 Nov. 2016
Developed for Python 3.5.1
Uses:
 - numpy 1.11.1
 - os, pickle
"""

#==============================================================================
#Importations
#==============================================================================

import os
import pickle
from numpy import savez

#==============================================================================
# Global variables
#==============================================================================

__pivy_directory__ = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
__parent_directory__ = os.path.abspath(os.path.dirname(__pivy_directory__))
__data_directory__ = os.path.abspath(os.path.join(__parent_directory__, 'data'))


#==============================================================================
# Functions
#==============================================================================

def folder_check(folder_abs_path):
    """
    Check if the given path indicates a folder, and creates it if not.
    """

    if not os.path.isdir(folder_abs_path):
        os.mkdir(folder_abs_path)


def folder_str(folder):
    """
    Check existence or create folder for a given string name or tuple of
    string name.
    """

    folder_path = __data_directory__
    folder_check(folder_path)

    if type(folder) != str:
        for elt in folder:
            folder_path += os.sep + elt
            folder_check(folder_path)
    else:
        folder_path += os.sep + folder
        folder_check(folder_path)

    return folder_path

def save_data_pickle(param_dict, name, folder):
    """
    Save data using pickle.
    """

    folder_path = folder_str(folder)
    full_path = folder_path + os.sep + name
    pickle.dump(param_dict, open(full_path, 'wb'))

def save_data_numpy(array_dict, name, folder):
    """
    Save data using numpy savez function.
    """

    folder_path = folder_str(folder)
    full_path = folder_path + os.sep + name + '.npz'
    savez(full_path, **array_dict)
