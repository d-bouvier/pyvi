# -*- coding: utf-8 -*-
"""
Test script for pyvi.utilities.savebox

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 19 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from pyvi.utilities.savebox import (create_folder, save_data, load_data,
                                    save_figure)


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    print('')

    filepath_abs = os.path.dirname(__file__)

    ##############################
    ## Function create_folder() ##
    ##############################

    print('Testing create_folder()...', end=' ')
    path1 = create_folder('a')
    path2 = create_folder('b', abs_path=filepath_abs)
    path3 = create_folder(['c', 'c1'])
    path4 = create_folder(['c', 'c2'])
    print('Done.')


    ##########################
    ## Function save_data() ##
    ##########################

    print('Testing save_data()...', end=' ')
    data_n = {'1': np.ones((1,)), '2': 2*np.ones((1,))}
    data_p = {'1': 1, '2': 2}
    save_data(data_n, 'test1', 'a', mode='numpy')
    save_data(data_n, 'test2', 'b', abs_path=filepath_abs)
    save_data(data_p, 'test3', ['c', 'c1'], mode='pickle')
    save_data(data_p, 'test4', path4, mode='pickle')
    print('Done.')


    ##########################
    ## Function load_data() ##
    ##########################

    print('Testing load_data()...', end=' ')
    data1 = load_data('test1', 'a')
    data2 = load_data('test2.npz', path2)
    data3 = load_data('test3.pyvi', ['c', 'c1'])
    data4 = load_data('test4', ['c', 'c2'], mode='pickle')
    for key, data in data_n.items():
        assert data1[key] == data, 'Problem with load_data() function.'
        assert data2[key] == data, 'Problem with load_data() function.'
    assert data3 == data_p, 'Problem with load_data() function.'
    assert data4 == data_p, 'Problem with load_data() function.'
    data1.close()
    data2.close()
    print('Done.')


    ############################
    ## Function save_figure() ##
    ############################

    print('Testing save_figure()...', end=' ')
    test = plt.figure('Test')
    plt.plot([1, 2, 3])
    save_figure(test, 'fig.png', 'figs')
    save_figure(test, 'fig.pdf', 'figs')
    plt.close(test)
    print('Done.')


    ##################################
    ## Cleaning working directory() ##
    ##################################

    print('Suppressing created files and folders...', end=' ')
    os.remove('a/test1.npz')
    os.remove(filepath_abs + os.sep + 'b/test2.npz')
    os.remove('c/c1/test3.pyvi')
    os.remove('c/c2/test4.pyvi')
    os.remove('figs/fig.png')
    os.remove('figs/fig.pdf')
    os.rmdir('a')
    os.rmdir(filepath_abs + os.sep + 'b')
    os.rmdir('c/c1')
    os.rmdir('c/c2')
    os.rmdir('c')
    os.rmdir('figs')
    print('Done.')