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


#==============================================================================
# Global variables
#==============================================================================

__data_directory__ = os.path.abspath(os.path.join(os.path.dirname( \
                        os.path.dirname(os.path.realpath(__file__))), 'data'))


