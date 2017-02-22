# -*- coding: utf-8 -*-
"""
Tools for combinatorics.

Multilinear combinatorics
-------------------------
make_list_pq :
    Compute the list of multilinear pq-functions used in each order of
    nonlinearity.
elimination :
    Eliminates the multilinear pq-functions unused in the system.
state_combinatorics :
    Compute, for each Mpq function at a given order n, the different sets of
    state-homogenous-order that are the inputs of the multilinear pq-function.
make_dict_pq_set :
    Return the list of sets characterising multilinear pq-functions used in a
    system.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 3 Nov. 2016
Developed for Python 3.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import itertools as itertbx
import numpy as np
from scipy.special import binom as binomial


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


def make_list_pq(nl_order_max):
    """
    Compute the list of multilinear pq-functions used in each order of
    nonlinearity.

    Parameters
    ----------
    nl_order_max : int
        Maximum order of nonlinearity

    Returns
    -------
    list_pq : numpy.ndarray
        Array of shape (N, 3), where N is the number of sets, each of the from
        [n, p, q].
    """

    # Initialisation
    list_pq = np.empty((0, 3), dtype=int)
    # Variable for reporting sets from the previous order
    nb_set_2_report = 0

    # Loop on order of nonlinearity
    for n in range(2, nl_order_max+1):
        # Report previous sets and change the corresponding order
        list_pq = np.concatenate((list_pq, list_pq[-nb_set_2_report-1:-1,:]))
        list_pq[-nb_set_2_report:,0] += 1
        # Loop on all new combination (p,q)
        for q in range(n+1):
            array_tmp = np.array([n, n-q, q])
            array_tmp.shape = (1, 3)
            list_pq = np.concatenate((list_pq, array_tmp))
            # We don't report the use of the pq-function for p = 0
            if not (n == q):
                nb_set_2_report += 1

    return list_pq


def elimination(h_pq_bool, list_pq):
    """
    Eliminates the multilinear pq-functions unused in the system.

    Parameters
    ----------
    h_pq_bool : function (int, int: boolean)
        Indicates, for given p & q, if the multilinear pq-function is used in
        the system.
    list_pq : numpy.ndarray
        Array of all combination [n, p, q].

    Outputs
    -------
    list_pq : numpy.ndarray
        Same array as the ``list_pq`` input array with unused lines deleted.
    """

    # Initialisation
    mask_pq = np.empty(list_pq.shape[0], dtype=bool)
    # Loop on all set combination
    for idx in range(list_pq.shape[0]):
        # In the following:
        # list_pq[idx,0] represents n
        # list_pq[idx,1] represents p
        # list_pq[idx,2] represents q
        mask_pq[idx] = h_pq_bool(list_pq[idx,1], list_pq[idx,2])

    return list_pq[mask_pq]


def state_combinatorics(list_pq, nl_order_max, sym_bool=False):
    """
    Compute, for each Mpq function at a given order n, the different sets of
    state-homogenous-order that are the inputs of the multilinear pq-function.
    All sets are created, even those identical in respect to the order (so, if
    the multilinear pq-function are symmetric, there is redudancy).

    Parameters
    ----------
    list_pq : numpy.ndarray
        Array of all combination [n, p, q].
    sym_bool : boolean, optional
        If True, does not computes sets that are equals given a shuffle.

    Outputs
    -------
    pq_sets : dict
        Dict of all tuple (p, q, k, nb) for each order n, where k are the sets
        state-homogenous-order, an nb the number of unordered sets equals to k,
        including k (equals to 1 if sym_bool is False).
    """

    # Initialisation
    pq_sets = {}
    for n in range(2, nl_order_max+1):
        pq_sets[n] = []

    for elt in list_pq:
        # In the following:
        # elt[0] represents n
        # elt[1] represents p
        # elt[2] represents q

        # Maximum value possible for a state order
        k_sum = elt[0] - elt[2]
        # Value needed for the sum of all state order
        k_max = k_sum - elt[1] + 1
        # Loop on all possible sets
        if sym_bool:
            list_idx = itertbx.combinations_with_replacement(range(1, k_max+1),
                                                             elt[1])
            list_idx_filtre = itertbx.filterfalse(lambda x: sum(x) != k_sum,
                                                  list_idx)
            for index in list_idx_filtre:
                nb_repetitions = 1
                current_max = 0
                for value in set(index):
                    nb_appearance = index.count(value)
                    current_max += nb_appearance
                    nb_repetitions *= int(binomial(current_max, nb_appearance))
                pq_sets[elt[0]].append((int(elt[1]), int(elt[2]), index,
                                        nb_repetitions))
        else:
            list_idx = itertbx.product(range(1, k_max+1), repeat=elt[1])
            list_idx_filtre = itertbx.filterfalse(lambda x: sum(x) != k_sum,
                                                  list_idx)
            for index in list_idx_filtre:
                pq_sets[elt[0]].append((elt[1], elt[2], index, 1))

    return pq_sets


def make_dict_pq_set(h_pq_bool, nl_order_max, sym_bool=False):
    """
    Return the list of sets characterising multilinear pq-functions used in a
    system.

    Parameters
    ----------
    h_pq_bool : function (int, int: boolean)
        Indicates, for given p & q, if the multilinear pq-function is used in
        the system.
    nl_order_max : int
        Maximum order of nonlinearity.
    print_opt : boolean, optional
        Intermediate results printing option.

    Returns
    -------
    mpq_sets : dict of lists
        Dict of keys n, each associated to a list of tuple (p, q, k, nb) with:
    - n : int
        Order of nonlinearity where the multilinear pq-function is used.
    - p : int
        Number of state-entry for the multilinear pq-function.
    - q : int
        Number of input-entry for the multilinear pq-function.
    - k : tuple (of length p)
        Homogenous orders for the state-entries.
    - nb : int
        Number of unordered sets equals to k, including k.

    """

    ## Main ##
    list_pq = make_list_pq(nl_order_max)
    list_pq = elimination(h_pq_bool, list_pq)
    pq_sets = state_combinatorics(list_pq, nl_order_max, sym_bool)

    return pq_sets


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    N = 5
    list_pq = make_list_pq(N)
    pq_sets = state_combinatorics(list_pq, N, True)
    for n, value in pq_sets.items():
        print(n)
        for elt in value:
            print(' ', elt)
