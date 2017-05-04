# -*- coding: utf-8 -*-
"""
Toolbox for plots.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 24 Apr. 2017
Developed for Python 3.6.1
"""

#==============================================================================
# Importations
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


#==============================================================================
# Functions
#==============================================================================

def plot_sig_io(input_sig, output_sig, time_vec, name=None,
                xlim=[None, None], ylim=[None, None]):
    complex_bool = 'complex' in str(input_sig.dtype) or \
                   'complex' in str(output_sig.dtype)
    nb_col = 2 if complex_bool else 1

    plt.figure(name)
    plt.clf()

    plt.subplot(2, nb_col, 1)
    plt.plot(time_vec, input_sig.real, 'b')
    plt.title('Input - Real part' if complex_bool else 'Input')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.subplot(2, nb_col, 3 if complex_bool else 2)
    plt.plot(time_vec, output_sig.real, 'b')
    plt.title('Output - Real part' if complex_bool else 'Output')
    plt.xlim(xlim)
    plt.ylim(ylim)
    if complex_bool:
        plt.subplot(2, nb_col, 2)
        plt.plot(time_vec, input_sig.imag, 'r')
        plt.title('Input - imaginary part')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.subplot(2, nb_col, 4)
        plt.plot(time_vec, output_sig.imag, 'r')
        plt.title('Output - imaginary part')
        plt.xlim(xlim)
        plt.ylim(ylim)
    plt.show()


def plot_sig_coll(sig_coll, time_vec, name=None, title_plots=None,
                  xlim=[None, None], ylim=[None, None], dim=1):
    nb_sig = sig_coll.shape[dim]
    complex_bool = 'complex' in str(sig_coll.dtype)
    if title_plots is None:
        title_plots = ['Signal {}'.format(n+1) for n in range(nb_sig)]

    plt.figure(name)
    plt.clf()

    if complex_bool:
        for n in range(nb_sig):
            plt.subplot(nb_sig, 2, 2*n+1)
            plt.plot(time_vec, sig_coll[:, n].real, 'b')
            plt.title(title_plots[n] + ' - real part')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.subplot(nb_sig, 2, 2*n+2)
            plt.plot(time_vec, sig_coll[:, n].imag, 'r')
            plt.title(title_plots[n] + ' - imaginary part')
            plt.xlim(xlim)
            plt.ylim(ylim)
    else:
        for n in range(nb_sig):
            plt.subplot(nb_sig, 1, n+1)
            plt.plot(time_vec, sig_coll[:, n], 'b')
            plt.title(title_plots[n])
            plt.xlim(xlim)
            plt.ylim(ylim)
    plt.show()


def plot_kernel_time(vec, kernel, style='wireframe', title=None,
                     linewidth=0.5, nb_levels=20):
    """
    Plots a discrete time kernel of order 1 or 2.

    Parameters
    ----------
    vec : numpy.ndarray
        Time vector.
    kernel : 1-D or 2-D numpy.ndarray
        Kernel to plot.
    style : {'surface', 'contour', 'wireframe'}, optional
        Plot mode if the kernel is of order 2.
    title : string, optional
        Title of the Figure.
    N : int, optional
        Optional parameter when using 'countour'
    """

    order = kernel.ndim

    if order ==1:
        if not title:
            title = 'Volterra kernel of order 1 (linear filter)'
        plt.figure(title)
        plt.clf()
        plt.plot(vec, kernel)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    elif order ==2:
        if not title:
            title = 'Volterra kernel of order 2'
        time_x, time_y = np.meshgrid(vec, vec)
        plt.figure(title)
        plt.clf()

        if style == 'contour':
            plt.contourf(time_x, time_y, kernel, nb_levels)
            plt.colorbar(extend='both')
            plt.xlabel('Time (s)')
            plt.ylabel('Time (s)')
        elif style == 'surface':
            ax = plt.subplot(111, projection='3d')
            surf = ax.plot_surface(time_x, time_y, kernel,
                                   antialiased=True, cmap='jet',
                                   rstride=1, cstride=1)
            plt.colorbar(surf, extend='both')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Time (s)')
            ax.set_zlabel('Amplitude')
        elif style == 'wireframe':
            ax = plt.subplot(111, projection='3d')
            ax.plot_wireframe(time_x, time_y, kernel, linewidth=linewidth,
                              antialiased=True, cmap='jet')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Time (s)')
            ax.set_zlabel('Amplitude')

    else:
        print('No plot possible, the kernel is of order {}.'.format(order))


def plot_kernel_freq(vec, kernel, style='wireframe', title=None,
                     db=True, unwrap_angle=True, logscale=False,
                     linewidth=0.5, nb_levels=20):
    """
    Plots a discrete time kernel of order 1 or 2.

    Parameters
    ----------
    vec : numpy.ndarray
        Frequency vector.
    kernel : 1-D or 2-D numpy.ndarray
        Kernel to plot.
    style : {'surface', 'contour', 'wireframe'}, optional
        Plot mode if the kernel is of order 2.
    title : string, optional
        Title of the Figure.
    N : int, optional
        Optional parameter when using 'countour'
    db : boolean, optional
        Choose wether or not magnitude is expressed in deciBel.
    unwrap_angle : boolen, optional
        Choose wether or not the phase is unwrapped.
    logscale: boolen or int, optional
        If False, all frequency axis are on a linear scale. If True, should be
        an int, and all frequency axis will be plotted using a logscale of base
        ``logscale``.
    """

    order = kernel.ndim
    kernel_amp = np.abs(kernel)
    kernel_phase = np.angle(kernel)
    amplabel = 'Magnitude'
    if db:
        kernel_amp = 20*np.log10(kernel_amp)
        amplabel += ' (dB)'
    if unwrap_angle:
        for n in range(order):
            kernel_phase = np.unwrap(kernel_phase, n)
    idx = slice(len(vec)//2,len(vec))

    if order ==1:
        if not title:
            title = 'Transfer kernel of order 1 (linear filter)'
        plt.figure(title)
        plt.clf()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        ax1.plot(vec[idx], kernel_amp[idx])
        ax2.plot(vec[idx], kernel_phase[idx])
        ax1.set_ylabel(amplabel)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (radians)')
        if logscale:
            ax1.set_xscale('log')
            ax2.set_xscale('log')
        ax1.set_xlim([0, vec[-1]])
        ax2.set_xlim([0, vec[-1]])

    elif order ==2:
        if not title:
            title = 'Trnsfer kernel of order 2'
        freq_x, freq_y = np.meshgrid(vec[idx], vec, indexing='ij')
        plt.figure(title)
        plt.clf()

        if style == 'contour':
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
            ax1.contourf(freq_x, freq_y, kernel_amp[idx,:], nb_levels)
            ax2.contourf(freq_x, freq_y, kernel_phase[idx,:], nb_levels)
        if style == 'surface':
            ax1 = plt.subplot(211, projection='3d')
            ax2 = plt.subplot(212, projection='3d')
            ax1.plot_surface(freq_x, freq_y, kernel_amp[idx,:],
                             antialiased=True, cmap='jet', rstride=1, cstride=1)
            ax2.plot_surface(freq_x, freq_y, kernel_phase[idx,:],
                             antialiased=True, cmap='jet', rstride=1, cstride=1)
        if style == 'wireframe':
            ax1 = plt.subplot(211, projection='3d')
            ax2 = plt.subplot(212, projection='3d')
            ax1.plot_wireframe(freq_x, freq_y, kernel_amp[idx,:],
                               linewidth=linewidth, antialiased=True)
            ax2.plot_wireframe(freq_x, freq_y, kernel_phase[idx,:],
                               linewidth=linewidth, antialiased=True)

        if style == 'contour':
            ax2.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel(amplabel)
            ax2.set_ylabel('Phase (radians)')
        else:
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Frequency (Hz)')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Frequency (Hz)')
            ax1.set_zlabel(amplabel)
            ax2.set_zlabel('Phase (radians)')
        if logscale:
            ax1.set_xscale('symlog')
            ax1.set_yscale('symlog')
            ax2.set_xscale('symlog')
            ax2.set_yscale('symlog')
        ax1.set_xlim([0, vec[-1]])
        ax2.set_xlim([0, vec[-1]])

    else:
        print('No plot possible, the kernel is of order {}.'.format(order))