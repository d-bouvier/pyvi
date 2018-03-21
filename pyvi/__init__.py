# -*- coding: utf-8 -*-
"""
Python toolbox for Volterra series identification using order separation.

Notes
-----
Developed for Python 3.6
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

from .__config__ import (__author__, __maintainer__, __version__)

from . import utilities
from . import volterra
from . import separation
from . import identification

__all__ = ['__author__', '__maintainer__', '__version__']
__all__ += ['utilities', 'volterra', 'separation', 'identification']
