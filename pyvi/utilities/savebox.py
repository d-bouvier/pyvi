# -*- coding: utf-8 -*-
"""
Tooolbox for saving data and figures.

Functions
---------
create_folder :
    Creates folder for given list of subfolders.; also returns folder path.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 28 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

from numpy import savez, load
import os
import pickle


#==============================================================================
# Global variables
#==============================================================================

__pivy_directory__ = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


#==============================================================================
# Functions
#==============================================================================

def _folder_check(folder_path):
    """
    Check if the given path indicates a folder, and creates it if not.

    Parameters
    ----------
    folder_path : str
        Path to check.
    """

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


def create_folder(folder_relative_path, abs_path=None):
    """
    Creates folder for given list of subfolders.; also returns folder path.

    Parameters
    ----------
    folder_relative_path : str or list(str)
        Absolute path or subfolder hierarchy where folder must be created.
    abs_path : {None, str}, optional (default=None)
        Absolute path where to begin relative path. If None, os.cwd() is used.

    Returns
    -------
    folder_abs_path : str
        Absolute path of the created folder.
    """

    if abs_path is None:
        folder_abs_path = os.getcwd()
    else:
        folder_abs_path = abs_path

    if type(folder_relative_path) != str:
        for elt in folder_relative_path:
            folder_abs_path += os.sep + elt
            _folder_check(folder_abs_path)
    else:
        folder_abs_path += os.sep + folder_relative_path
        _folder_check(folder_abs_path)

    return folder_abs_path


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


def save_figure(handle_fig, name, folder):
    """
    Save figure using matplotlib.
    """

    folder_path = folder_str(folder)
    full_path = folder_path + os.sep + name
    handle_fig.savefig(full_path, bbox_inches='tight')


def load_data_pickle(name, folder):
    """
    Load data using pickle.
    """

    folder_path = folder_str(folder)
    full_path = folder_path + os.sep + name
    return pickle.load(open(full_path, 'rb'))


def load_data_numpy(name, folder):
    """
    Load data using numpy savez function.
    """

    folder_path = folder_str(folder)
    full_path = folder_path + os.sep + name + '.npz'
    return load(full_path)