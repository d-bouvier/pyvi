# -*- coding: utf-8 -*-
"""
Tooolbox for saving data and figures.

Functions
---------
create_folder :
    Creates folder for given list of subfolders.; also returns folder path.
save_data :
    Save data using either numpy.savez() or pickle.dump().
load_data :
    Load data using either numpy.load() or pickle.load().
save_figure :
    Save figure using matplotlib.

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

import os
import pickle
import numpy as np


#==============================================================================
# Global variables
#==============================================================================

__pivy_directory__ = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
__numpy_extension__ = '.npz'
__pickle_extension__ = '.pyvi'


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


def _check_path(folder_path, abs_path=None):
    """
    Checks if ``folder_path``is an absolute path or a subfolder hierarchy.

    Parameters
    ----------
    folder_path : str or list(str)
        Absolute path or subfolder hierarchy where file must be created.
    abs_path : {None, str}, optional (default=None)
        Absolute path where to begin relative path. If None, os.cwd() is used.
    """

    if (type(folder_path) != str) or (not os.path.isabs(folder_path)):
        return create_folder(folder_path, abs_path=abs_path)
    else:
        return folder_path


def save_data(data_dict, name, folder_path, mode='numpy', abs_path=None):
    """
    Save data using either numpy.savez() or pickle.dump().

    Parameters
    ----------
    data_dict : dict(str: all_types)
        Dictionary of data to save.
    name: str
        Name of file to create.
    folder_path: str or list(str)
        Absolute path or subfolder hierarchy where the file will be created.
    mode: {'numpy', 'pickle'}, optional (default='numpy')
        Defines which function is used: numpy.savez() or pickle.dump().
    abs_path: {None, str}, optional (default=None)
        Absolute path where to begin relative path. If None, os.cwd() is used.
    """

    full_path = _check_path(folder_path, abs_path=abs_path) + os.sep + name

    if mode == 'numpy':
        full_path += __numpy_extension__
        np.savez(full_path, **data_dict)
    elif mode == 'pickle':
        full_path += __pickle_extension__
        pickle.dump(data_dict, open(full_path, 'wb'))


def load_data(name, folder_path, mode=None, abs_path=None):
    """
    Load data using either numpy.load() or pickle.load().

    Parameters
    ----------
    name: str
        Name of file to load.
    folder_path: str or list(str)
        Absolute path or subfolder hierarchy where the file is located.
    mode: {'numpy', 'pickle' or None}, optional (default=None)
        Format used when saving data. If None, found from from file extension.
    abs_path: {None, str}, optional (default=None)
        Absolute path where to begin relative path. If None, os.cwd() is used.
    """

    basename, ext = os.path.splitext(name)
    full_path = _check_path(folder_path, abs_path=abs_path) + os.sep + basename

    _full_numpy_path = full_path + __numpy_extension__
    _full_pickle_path = full_path + __pickle_extension__
    _has_numpy_version = os.path.isfile(_full_numpy_path)
    _has_pickle_version = os.path.isfile(_full_pickle_path)

    if ext == '':
        if mode == 'numpy':
            assert _has_numpy_version, 'No file {} found with'.format(name) + \
                ' with specified extension {} '.format(__numpy_extension__) + \
                '(full search path is {}).'.format(_full_numpy_path)
            full_path = _full_numpy_path
        if mode == 'pickle':
            assert _has_pickle_version, 'No file {} found with'.format(name) + \
                ' specified extension {} '.format(__pickle_extension__) + \
                '(full search path is {}).'.format(_full_pickle_path)
            full_path = _full_pickle_path
        if mode is None:
            assert not (_has_numpy_version and _has_pickle_version), \
                'Two files {} found with extension '.format(name) + \
                '{} and {}'.format(__numpy_extension__, __pickle_extension__) +\
                '(full search path is {})'.format(full_path)
            assert not (not _has_numpy_version and not _has_pickle_version), \
                'No files {} found with extension '.format(name) + \
                '{} or {}'.format(__numpy_extension__, __pickle_extension__) + \
                '(full search path is {})'.format(full_path)
            if _has_numpy_version:
                mode = 'numpy'
                full_path = _full_numpy_path
            if _has_pickle_version:
                mode = 'pickle'
                full_path = _full_pickle_path
    elif ext == __numpy_extension__:
        mode = 'numpy'
        full_path = _full_numpy_path
    elif ext == __pickle_extension__:
        mode = 'pickle'
        full_path = _full_pickle_path

    if mode == 'numpy':
        return np.load(full_path)
    elif mode == 'pickle':
        return pickle.load(open(full_path, 'rb'))


def save_figure(handle_fig, name, folder_path, abs_path=None):
    """
    Save figure using matplotlib.

    Parameters
    ----------
    handle_fig :
        Handle of the figure to save.
    name : str
        Name of file to load.
    folder_path : str or list(str)
        Absolute path or subfolder hierarchy where the file is located.
    abs_path : {None, str}, optional (default=None)
        Absolute path where to begin relative path. If None, os.cwd() is used.
    """

    full_path = _check_path(folder_path, abs_path=abs_path) + os.sep + name
    handle_fig.savefig(full_path, bbox_inches='tight')