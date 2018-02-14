# -*- coding: utf-8 -*-
"""
Module for miscellanous functions or classes.

Notes
-----
Developed for Python 3.6.1
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""


#==============================================================================
# Class
#==============================================================================

class Style:
    """
    Class for color in terminal.
    """

    RESET = '\033[0m'
    BRIGHT = '\033[1m'
    BOLD = '\033[2m'
    UNDERLINE = '\033[4m'
    ITALIC = '\033[3m'

    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    BBLACK = '\033[40m'
    BRED = '\033[41m'
    BGREEN = '\033[42m'
    BYELLOW = '\033[43m'
    BBLUE = '\033[44m'
    BPURPLE = '\033[45m'
    BCYAN = '\033[46m'
    BWHITE = '\033[47m'
