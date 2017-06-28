# -*- coding: utf-8 -*-
"""
Test script for and pyvi.simulation.combinatorics

Notes
-----
@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 27 June 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import pyvi.simulation.combinatorics as comb


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    N = 4
    pq_dict = {(2, 0): 1,
               (1, 1): 1}

    def print_sets(sets):
        for n, value in pq_sets.items():
            print(n)
            for elt in value:
                print(' ', elt)

    list_pq = comb.make_list_pq(N)
    print('pq-list\n-------', *list_pq, sep='\n')
    pq_sets = comb.state_combinatorics(list_pq, N, True)
    print('pq-sets\n-------')
    print_sets(pq_sets)

    list_pq = comb.elimination(pq_dict, list_pq)
    print('pq-list (filtered)\n-----------------', *list_pq, sep='\n')
    pq_sets = comb.state_combinatorics(list_pq, N, True)
    print('pq-sets (filtered)\n-----------------')
    print_sets(pq_sets)