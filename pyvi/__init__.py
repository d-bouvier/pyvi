# -*- coding: utf-8 -*-
"""
Package for simulation and analysis of nonlinear system.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""

from .__config__ import (__author__, __maintainer__, __version__,
                         __dependencies__)

from . import utilities
from . import separation as sep
from . import identification as identif
