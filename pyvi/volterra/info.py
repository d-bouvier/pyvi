# -*- coding: utf-8 -*-
"""
Module creating various tools for Volterra series.

Number of coefficients and kernel form (see :mod:`pyvi.volterra.tools`)
-----------------------------------------------------------------------
kernel_nb_coeff :
    Returns the meaningful number coefficients in a Volterra kernel.
series_nb_coeff :
    Returns the meaningful number of coefficients in a Volterra series.
vec2kernel :
    Rearranges a vector of Volterra coefficients of order n into a tensor.
vec2series :
    Rearranges a vector of all Volterra coefficients into a dict of tensors.
kernel2vec :
    Rearranges a Volterra kernel from tensor shape to vector form.

Combinatorial basis (see :mod:`pyvi.volterra.combinatorial_basis`)
------------------------------------------------------------------
compute_combinatorial_basis :
    Creates dictionary of combinatorial basis matrix.

Developed for Python 3.6
@author: Damien Bouvier (Damien.Bouvier@ircam.fr)
"""
