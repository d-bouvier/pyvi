# -*- coding: utf-8 -*-
"""
Toolbox for nonlinear system identification.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 24 Mar. 2016
Developed for Python 3.5.1
Uses:
 - numpy 1.11.1
 - scipy 0.18.0
 - pyvi 0.1
 - itertools
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import binom as binomial
import itertools as itr
from pyvi.simulation.simulation import simulation
from pyvi.tools.mathbox import rms, safe_db


#==============================================================================
# Functions
#==============================================================================

def identification(input_sig, output_sig, M=1, order_max=1):
    """
    Identify the Volterra kernels of a system from input and output signals.

    Parameters
    ----------
    input_sig : numpy.ndarray
        Vector of input signal.
    output_sig : numpy.ndarray
        Vector of output signal.
    M : int
        Memory length of kernels
    order_max : int, optional
        Highest kernel order (default 1).

    Returns
    -------
    kernels : dict of numpy.ndarray
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    # Input combinatoric
    phi_m = construct_phi_matrix(input_sig, M, order_max)

    # QR decomposition
    q, r = np.linalg.qr(phi_m)

    # Forward inverse
    y = np.dot(q.T, output_sig)
    f = solve_triangular(r, y)

    # Re-arranging vector f into volterra kernels
    kernels = vector_to_kernels(f, M, order_max)

    return kernels


def construct_phi_matrix(signal, M, order_max=1):
    """
    Construct the Volterra basis functionals for kernels up to a certain order,
    of memory ``M``, and for a given input signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Vector of input signal.
    M : int
        Memory length of kernels.
    order_max : int, optional
        Highest kernel order (default 1).

    Returns
    -------
    phi : numpy.ndarray
        Matrix containing the expression of the Volterra basis functionals for
        all orders (up to ``order_max``) and all samples of ``signal``.
    """

    # Initialization
    nb_terms = int(binomial(M + order_max, order_max)) - 1
    phi = np.zeros((nb_terms, signal.shape[0]))
    padded_signal = np.pad(signal, (M-1, 0), 'constant')
    list_ind = np.arange(M-1, padded_signal.shape[0])

    # Main loop
    ind = 0
    for n in range(1, order_max+1):
        product = construct_volterra_basis_functionals(padded_signal, M, n)
        ind_iterator = itr.combinations_with_replacement(range(M), n)
        for indexes in ind_iterator:
            temp_arg = tuple()
            for iii in reversed(indexes):
                temp_arg += (list_ind-iii,)
            phi[ind] = product[temp_arg]
            ind += 1

    return phi.T


def construct_volterra_basis_functionals(signal, M, n):
    """
    Construct the Volterra basis functionals for a kernel of order ``n``, of
    memory ``M``, and for a given input signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Vector of input signal.
    M : int
        Memory length of kernels.
    order : int
        Kernel order.

    Returns
    -------
    product_array : numpy.ndarray
        Muldimensional array with the basis functionals in the main diagonals.
    """

    # Initialization
    length = signal.shape[0]
    list_ind = np.arange(M-1, length)
    iterator = itr.combinations_with_replacement(range(M), n-1)
    product_array = np.zeros((length,) * n)

    # Main loop
    for indexes in iterator:
        temp_arg = tuple()
        for iii in reversed(indexes):
            temp_arg += (list_ind-iii,)
        temp_arg += (list_ind,)
        product_array[temp_arg] = np.prod(signal[np.stack(temp_arg)], axis=0)

    return product_array


def vector_to_kernels(f, M, order_max=1):
    """
    Rearrange a numpy vector containing the coefficients of all Volterra kernels
    up to order ``order_max`` into a dictionnary regrouping numpy.ndarray
    representing the Volterra kernels.

    Parameters
    ----------
    f : numpy.ndarray
        Vector regrouping all coefficients.
    M : int
        Memory length of kernels.
    order_max : int, optional
        Highest kernel order (default 1).

    Returns
    -------
    kernels : dict of numpy.ndarray
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    # Check dimension
    length = int(binomial(M + Nmax, Nmax)) - 1
    assert f.shape[0] == length, \
           'The vector of Volterra coefficients has wrong length ' + \
           '(got {}, expected {}).'.format(f.shape[0], length)

    # Initialization
    kernels = dict()
    current_ind = 0

    # Loop on all orders of nonlinearity
    for n in range(1, Nmax+1):
        nb_term = int(binomial(M + n - 1, n))
        kernels[n] = vector_to_kernel(f[current_ind:current_ind+nb_term], M, n)
        current_ind += nb_term

    return kernels


def vector_to_kernel(vec, M, order):
    """
    Rearrange a numpy vector containing the coefficients of a Volterra kernel of
    order ``n`` into a numpy.ndarray representing the Volterra kernel.

    Parameters
    ----------
    vec : numpy.ndarray
        Vector regrouping all coefficients of the kernel.
    M : int
        Memory length of the kernel.
    order : int
        Kernel order.

    Returns
    -------
    kernel : numpy.ndarray
        The coefficients arranged as a Volterra kernel.
    """

    # Check dimension
    length = int(binomial(M + order - 1, order))
    assert vec.shape[0] == length, 'The vector of coefficients for ' + \
            'Volterra kernel of order {} has wrong length'.format(order) + \
            '(got {}, expected {}).'.format(vec.shape[0], length)

    # Initialization
    kernel = np.zeros((M,)*order)
    current_ind = 0

    # Loop on all combinations for order n
    for indexes in itr.combinations_with_replacement(range(M), order):
        kernel[indexes] = vec[current_ind]
        current_ind += 1

    return kernel


def error_measure(estimated_kernels, true_kernels, db=True):
    """
    Give the measurment error of the kernel identification, as the RMS value of
    the difference divided by the RMS of the trus kernels (for each order).

    Parameters
    ----------
    estimated_kernels : dict of numpy.ndarray
        Dictionnary of kernels values (estimation).
    true_kernels : dict of numpy.ndarray
        Dictionnary of kernels values (ground truth).

    Returns
    -------
    error : list of floats
        List of normalized-RMS error values.
    """

    # Initialization
    errors = []

    # Loop on all estimated kernels
    for n, kernel in estimated_kernels.items():
        num = rms(kernel - true_kernels[n])
        den = rms(true_kernels[n])
        if db:
            errors.append(safe_db(num, den))
        else:
            errors.append(num/den)

    return errors


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    from pyvi.simulation.systems import second_order_w_nl_damping
    from pyvi.tools.plotbox import plot_kernel_time

    # System specification
    system = second_order_w_nl_damping(gain=1, f0=100,
                                       damping=0.2,
                                       nl_coeff=[1e-1, 3e-5])

    # Input signal specification
    fs = 2000
    T = 5
    Nmax = 2
    M = 60
    time_vector = np.arange(0, T, 1/fs)
    K = time_vector.shape[0]

    sigma = 1
    input_sig = np.random.normal(scale=sigma, size=K)

    # Simulation
    out_sig = simulation(input_sig, system, fs=fs, nl_order_max=Nmax,
                     hold_opt=0)

    # Identification
    kernels = identification(input_sig, out_sig, order_max=Nmax, M=M)
    kernels[2] = (1/2) * (kernels[2] + kernels[2].T)

    # Ground truth
    system.compute_volterra_kernels(fs, (M-1)/fs, order_max=Nmax, which='time')
    tau_vec = system._time_vector

    # Estimation error
    error = error_measure(kernels, system.volterra_kernels)
    print(error)

    # Plots
    system.compute_volterra_kernels(fs, (M-1)/fs, order_max=Nmax, which='time')
    tau_vec = system._time_vector
    kernels[2] = (1/2) * (kernels[2] + kernels[2].T)
#    style2D = 'wireframe'
    style2D = 'surface'

    plot_kernel_time(tau_vec, kernels[1],
                     title='Kernel of order 1 - Estimation')
    plot_kernel_time(tau_vec, system.volterra_kernels[1],
                     title='Kernel of order 1 - Ground truth')
    plot_kernel_time(tau_vec, kernels[2], style=style2D,
                     title='Kernel of order 2 - Estimation')
    plot_kernel_time(tau_vec, system.volterra_kernels[2], style=style2D,
                     title='Kernel of order 2 - Ground truth')


