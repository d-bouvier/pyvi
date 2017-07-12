# -*- coding: utf-8 -*-
"""
Tools for combinatorics.

Functions
---------
make_list_pq :
    Returns the list of all pq-functions used in each order of nonlinearity.
elimination :
    Eliminates the pq-functions unused in the system.
state_combinatorics :
    Computes, for each pq-function at a given order n, the different sets of
    state-homogenous-order that are the inputs of the multilinear pq-function.
make_pq_combinatorics :
    Returns the list of sets characterising pq-functions used in a system.

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 12 July 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from itertools import combinations_with_replacement, filterfalse, product
from ..utilities.mathbox import binomial


#==============================================================================
# Functions
#==============================================================================


def make_list_pq(nl_order_max):
    """
    Returns the list of all pq-functions used in each order of nonlinearity.

    Parameters
    ----------
    nl_order_max : int
        Maximum order of nonlinearity.

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


def elimination(pq_dict, list_pq):
    """
    Eliminates the pq-functions unused in the system.

    Parameters
    ----------
    pq_dict : dict((int, int): numpy.ndarray)
        Dictionnary listing all the tensor representing the pq-functions of the
        system.
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
        mask_pq[idx] = (list_pq[idx,1], list_pq[idx,2]) in pq_dict.keys()

    return list_pq[mask_pq]


def state_combinatorics(list_pq, nl_order_max, sym_bool=False):
    """
    Retuns the list of sets characterising pq-functions used in a system.

    Computes, for each pq-function at a given order n, the different sets of
    state-homogenous-order that are the inputs of the multilinear pq-function.

    Parameters
    ----------
    list_pq : numpy.ndarray
        Array of all combination [n, p, q].
    sym_bool : boolean, optional (default=False)
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
            list_idx = combinations_with_replacement(range(1, k_max+1), elt[1])
            list_idx_filtre = filterfalse(lambda x: sum(x) != k_sum, list_idx)
            for index in list_idx_filtre:
                nb_repetitions = 1
                current_max = 0
                for value in set(index):
                    nb_appearance = index.count(value)
                    current_max += nb_appearance
                    nb_repetitions *= binomial(current_max, nb_appearance)
                pq_sets[elt[0]].append((int(elt[1]), int(elt[2]), index,
                                        nb_repetitions))
        else:
            list_idx = product(range(1, k_max+1), repeat=elt[1])
            list_idx_filtre = filterfalse(lambda x: sum(x) != k_sum, list_idx)
            for index in list_idx_filtre:
                pq_sets[elt[0]].append((elt[1], elt[2], index, 1))

    return pq_sets


def make_pq_combinatorics(pq_dict, nl_order_max, sym_bool=False):
    """
    Returns the list of sets characterising pq-functions used in a system.

    Parameters
    ----------
    pq_dict : dict((int, int): numpy.ndarray)
        Dictionnary listing all the tensor representing the pq-functions of the
        system.
    nl_order_max : int
        Maximum order of nonlinearity.
    sym_bool : boolean (default=False)
        True if the pq-functions are under a symmetric form.

    Returns
    -------
    pq_comb : dict(int: list(tuple(int, int, int, int)))
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

    list_pq = make_list_pq(nl_order_max)
    list_pq = elimination(pq_dict, list_pq)
    pq_comb = state_combinatorics(list_pq, nl_order_max, sym_bool)

    return pq_comb