# -*- coding: utf-8 -*-
"""
Module for Volterra systems.

This package creates classes that allows use of the Volterra representation of
a system (see https://en.wikipedia.org/wiki/Volterra_series).

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 12 Sept. 2016
Developed for Python 3.5.1
Uses:
 - sympy 1.0
"""

#==============================================================================
# Importations
#==============================================================================

import sympy as sp
import sympy.combinatorics as spP
from abc import abstractmethod


#==============================================================================
# Class
#==============================================================================

class Kernel:
    """Symbolic representation of a transfer Volterra kernel."""

    def __init__(self, expression=sp.zeros(1), order=1, **kwargs):
        """Initialize the kernel with its formula and order.

        If neither the expression or the order are given, it is supposed that
        it is a null kernel of order 1."""

        self.expr = expression
        self.order = order
        self.dim_input = self.expr.shape[1]
        self.dim_output = self.expr.shape[0]

        self.symmetric = kwargs.get('symmetric', None)
        self.symb_name = kwargs.get('name',
                                    sp.Function('H{}'.format(self.order)))
        self.symb_var = kwargs.get('var', None)


    def __repr__(self):
        """Represents the kernel as its Sympy expression."""
        return sp.srepr(self.expr)


    def __str__(self):
        """Print the kernel expression using Sympy pretty printing."""
        return sp.pretty(self.symb_name(*self.symb_var)) + ' = \n' + \
               sp.pretty(self.expr)

    #=============================================#

    @abstractmethod
    def plot(self):
        """Plots kernels of order 1 and 2."""
        return NotImplementedError


    @abstractmethod
    def symmetrize(self):
        """Put the kernel expression into its symmetric form."""
        ### To optimize ###
        def kernel_expr(expr, var_old, var_new):
            dict_subs = {}
            for idx, val in enumerate(var_old):
                dict_subs[val] = var_new[idx]
            return expr.subs(dict_subs, simultaneous=True)

        result_tmp = sp.zeros(self.dim_output, self.dim_input)
        perm = spP.Permutation(range(self.order))
        for idx in range(perm.cardinality):
            result_tmp += kernel_expr(self.expr, self.symb_var,
                                      (perm + idx)(self.symb_var))
        self.expr = result_tmp/perm.cardinality
        self.symmetric = True


    @abstractmethod
    def regularize(self):
        """Put the kernel expression into its regular form."""
        raise NotImplementedError


    @abstractmethod
    def triangularize(self):
        """Put the kernel expression into its triangular form."""
        raise NotImplementedError



class Volterra:
    """SYmbolic representation of a Volterra serie of a system."""

    def __init__(self, kernels=[Kernel()], **kwargs):
        """Initialize the Volterra series.

        If no arguments are given, the Volterra serie is limited to a null
        kernel of order 1."""
        self.kernels = kernels
        self.list_kernels = []
        for idx, kernel in enumerate(self.kernels):
            self.list_kernels.append(kernel.order)

        self.order_max = kwargs.get('order_max', max(self.list_kernels))
        self.symmetric = kwargs.get('symmetric')
        if self.symmetric is None:
            self.symmetric = True
            for idx, kernel in enumerate(self.kernels):
                self.symmetric = self.symmetric and kernel.symmetric


    def __repr__(self):
        repr_str = '{\n'
        for name, attribute in self.__dict__.items():
            repr_str += name + ' : ' + attribute.__repr__() + '\n'
        return repr_str + '}'


    def __str__(self):
        print_str = 'Volterra serie up to order {}:\n'.format(self.order_max)
        for kernel in self.kernels:
            print_str += kernel.__str__() + '\n'
        return print_str

    #=============================================#

    @abstractmethod
    def plot(self):
        """Plots kernels of order 1 and 2."""
        return NotImplementedError


    @abstractmethod
    def symmetrize(self):
        """Put all kernel expression into their symmetric form."""
        for idx, kernel in enumerate(self.kernels):
            self.list_kernels[idx] = kernel.symmetrize()
        self.symmetric = True


    @abstractmethod
    def regularize(self):
        """Put all kernel expression into their regular form."""
        for idx, kernel in enumerate(self.kernels):
            self.list_kernels[idx] = kernel.regularize()


    @abstractmethod
    def triangularize(self):
        """Put all kernel expression into their triangular form."""
        for idx, kernel in enumerate(self.kernels):
            self.list_kernels[idx] = kernel.triangularize()


    @abstractmethod
    def inverse(self):
        """Gives the inverse Volterra series of the system."""
        return NotImplementedError

