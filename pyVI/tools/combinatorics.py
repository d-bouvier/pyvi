# -*- coding: utf-8 -*-
"""
Tools for combinatorics.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 12 Sept. 2016
Developed for Python 3.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import itertools as itertbx


#==============================================================================
# Functions
#==============================================================================

def multilinear_combinatorics(used_pq={}, order_max=3):
    """Returns combinatorics for a given set of multilinear functions.
    
    This function takes a set of (p, q) values (p and q being integers) and an
    optionnal maximum order, and returns a dictionary 'list_orders' structured
    as follows:
    - 'list_orders' keys are nonlinear order, and the corresponging value is
       a dictionnary 'pq_by_order';
    - 'pq_by_order' keys are tuple (p, q), and the corresponding value is the
       set of all possible combination for the order of the state-input of the
       corresponding multilinear function."""
       
    dict_pq_by_order = dict()
    dict_sets = dict()
    list_orders = [1]
    
    # Compute, for each order of nonlinearity, the set of (p, q) functions
    for n in range(2, order_max+1):
        # Take the (p, q) functions of previous order (if it exists)
        dict_pq_by_order[n] = dict_pq_by_order.get(n-1, set()).copy()
        # Discard the (0, n-1) function for the order n
        dict_pq_by_order[n].discard((0, n-1))
        # Add the new (p, q) functions (such that p+q=n)
        dict_pq_by_order[n].update({(n-q, q) for q in range(n+1)} & used_pq)
        # If there is no (p, q) function at this order, discard the key 'n'
        if len(dict_pq_by_order[n]) == 0:
            del dict_pq_by_order[n]
    
    # Compute, for a given order and (p, q) function, the set of possible input
    for n in sorted(dict_pq_by_order):
        dict_sets[n] = dict()
        # Loop on all (p, q) functions for this order
        for (p, q) in dict_pq_by_order[n]:
            dict_sets[n][(p, q)] = set()
            # Loop on all permutations with repetitions of p elements taken
            # from all inferior nonlinear orders
            for possible_set in itertbx.product(list_orders, repeat=p):
                # Discard all those not giving homogeneous order n
                if sum(possible_set) == n-q:
                    dict_sets[n][(p, q)].add(possible_set)
            # Discard this (p, q) function if no possible inputs were found
            if len(dict_sets[n][(p, q)]) == 0:
                del dict_sets[n][(p, q)]
        # Discard this order if no (p, q) function is used
        if len(dict_sets[n]) == 0:
            del dict_sets[n]
        else:
            list_orders.append(n)
    
    return dict_sets

