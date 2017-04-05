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
 - math
"""

#==============================================================================
# Importations
#==============================================================================

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import binom as binomial
import itertools as itr
from pyvi.tools.mathbox import rms, safe_db
from math import factorial


#==============================================================================
# Functions
#==============================================================================

def identification(input_sig, output_sig, M=1, order_max=1,
                   separated_orders=False, phi_m=None):
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
    separated_orders : boolean, optional
        If True, ``output_sig`` should contain the separated homogeneous order
        of the output, and the identification will be made for each kernel
        separately.

    Returns
    -------
    kernels : dict of numpy.ndarray
        Dictionnary linking the Volterra kernel of order ``n`` to key ``n``.
    """

    # Input combinatoric
    if phi_m == None:
        phi_m = construct_phi_matrix(input_sig, M, order_max)

    if separated_orders:
        f = np.array([])
        ind_start = 0

        for n in range(order_max):
            # Number of combination term
            nb_term = int(binomial(M + n, n + 1))

            # QR decomposition
            q_n, r_n = np.linalg.qr(phi_m[:,ind_start:ind_start+nb_term])
            print('Condition number order {}:'.format(n+1), np.linalg.cond(r_n))

            # Forward inverse
            current_y = np.dot(q_n.T, output_sig[n])
            f_n = solve_triangular(r_n, current_y)

            # Save this order kernel coefficient
            f = np.concatenate((f, f_n), axis=0)
            ind_start += nb_term

        # Re-arranging vector f into volterra kernels
        kernels = vector_to_kernels(f, M, order_max)

    else:
        # QR decomposition
        q, r = np.linalg.qr(phi_m)
        print('Condition number (total):', np.linalg.cond(r))

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
    #TODO optimiser calcul (si possible)

    # Initialization
    nb_terms = int(binomial(M + order_max, order_max)) - 1
    phi = np.zeros((nb_terms, signal.shape[0]))
    padded_signal = np.pad(signal, (M-1, 0), 'constant')
    list_ind = np.arange(M-1, padded_signal.shape[0])

    # Main loop
    ind = 0
    for n in range(1, order_max+1):
        product = volterra_basis_functionals(padded_signal, M, n)
        ind_iterator = itr.combinations_with_replacement(range(M), n)
        for indexes in ind_iterator:
            temp_arg = tuple()
            for iii in reversed(indexes):
                temp_arg += (list_ind-iii,)
            phi[ind] = product[temp_arg]
            ind += 1

    return phi.T


def volterra_basis_functionals(signal, M, n):
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
    #TODO optimiser calcul (si possible)

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

    return symmetrization(kernel)


def symmetrization(array):
    """
    Symmetrize a multidimensional square array (each dimension must have the
    same length).

    Parameters
    ----------
    array : numpy.ndarray
        Array to symmetrize.

    Returns
    -------
    array_sym : numpy.ndarray
        Symmetrized array.
    """

    shape = array.shape
    assert len(set(shape)) == 1, 'Multidimensional array is not square ' + \
        '(has shape {})'.format(shape)
    n = len(array.shape)

    array_sym = np.zeros(shape, dtype=array.dtype)
    for ind in itr.permutations(range(n), n):
        array_sym += np.transpose(array, ind)
    return array_sym / factorial(n)


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
    from pyvi.simulation.simulation import simulation
    from pyvi.tools.plotbox import plot_kernel_time
    import pyvi.tools.order_separation as sep

    # System specification
    system = second_order_w_nl_damping(gain=1, f0=100,
                                       damping=0.2,
                                       nl_coeff=[1e-1, 3e-5])

    # Input signal specification
    fs = 1000
    T = 2
    Nmax = 3
    M = 20
    time_vector = np.arange(0, T, 1/fs)
    K = time_vector.shape[0]

    assert (binomial(M+Nmax, Nmax) - 1) <= K

    sigma = np.sqrt(2)
    covariance = [[sigma, 0], [0, sigma]]
    random_sig = np.random.multivariate_normal([0, 0], covariance, size=K)
    input_sig_cplx = random_sig[:, 0] + 1j * random_sig[:, 1]
    input_sig = 2 * np.real(input_sig_cplx)


    ## Simulation and identification ##

    # Identification on output signal
    out_sig_1 = simulation(input_sig, system, fs=fs, nl_order_max=Nmax,
                           hold_opt=0)
    phi_1 = construct_phi_matrix(input_sig, M, order_max=Nmax)
    kernels_1 = identification(input_sig, out_sig_1, order_max=Nmax, M=M,
                               phi_m=phi_1)

    # Identification on separated orders (ground truth)
    out_sig_2 = simulation(input_sig, system, fs=fs, nl_order_max=Nmax,
                           out='output_by_order', hold_opt=0)
    kernels_2 = identification(input_sig, out_sig_2, order_max=Nmax, M=M,
                               separated_orders=True, phi_m=phi_1)

    # Identification on separated orders (phase+amp method)
    data_3, param_3 = sep.simu_collection(input_sig_cplx, system, fs=fs,
                                          hold_opt=0, name='sep',
                                          method='phase+amp',
                                          param={'nl_order_max' :Nmax,
                                                 'gain': 0.6,
                                                 'output': 'orders'})
    out_sig_3 = sep.order_separation(data_3['output_collection'],
                                     param_3['sep_method'],
                                     param_3['sep_param'])
    kernels_3 = identification(data_3['input_collection'][0], out_sig_3,
                               order_max=Nmax, M=M, separated_orders=True,
                               phi_m=phi_1)


    # Ground truth
    system.compute_volterra_kernels(fs, (M-1)/fs, order_max=Nmax, which='time')

    # Estimation error
    errors_1 = error_measure(kernels_1, system.volterra_kernels)
    errors_2 = error_measure(kernels_2, system.volterra_kernels)
    errors_3 = error_measure(kernels_3, system.volterra_kernels)
    print(errors_1)
    print(errors_2)
    print(errors_3)

    # Plots
    tau_vec = system._time_vector
    style2D = 'surface' # 'wireframe'
    str1 = ['Kernel of order 1 - ',  'Kernel of order 2 - ']
    str2 = ['Ground truth', 'Identification',
            'Identification on true separated orders',
            'Identification on estimated separated orders']

    plot_kernel_time(tau_vec, system.volterra_kernels[1],
                     title=str1[0]+str2[0])
    plot_kernel_time(tau_vec, system.volterra_kernels[2], style=style2D,
                     title=str1[1]+str2[0])

    plot_kernel_time(tau_vec, kernels_1[1],
                     title=str1[0]+str2[1])
    plot_kernel_time(tau_vec, kernels_1[2], style=style2D,
                     title=str1[1]+str2[1])

    plot_kernel_time(tau_vec, kernels_2[1],
                     title=str1[0]+str2[2])
    plot_kernel_time(tau_vec, kernels_2[2], style=style2D,
                     title=str1[1]+str2[2])

    plot_kernel_time(tau_vec, kernels_3[1],
                     title=str1[0]+str2[3])
    plot_kernel_time(tau_vec, kernels_3[2], style=style2D,
                     title=str1[1]+str2[3])
