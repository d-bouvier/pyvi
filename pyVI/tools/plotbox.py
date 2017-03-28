# -*- coding: utf-8 -*-
"""
Toolbox for plots.

@author: bouvier (bouvier@ircam.fr)
         Damien Bouvier, IRCAM, Paris

Last modified on 04 Oct. 2016
Developed for Python 3.5.1
Uses:
 - matplolib 1.5.1
"""

#==============================================================================
# Importations
#==============================================================================

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


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


def plot_kernel_time(vec, kernel, style='surface', title=None, N=20):
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
            plt.contourf(time_x, time_y, kernel, N)
            plt.colorbar(extend='both')
            plt.xlabel('Time (s)')
            plt.ylabel('Time (s)')
        elif style == 'surface':
            ax = plt.subplot(111, projection='3d')
            surf = ax.plot_surface(time_x, time_y, kernel, linewidth=0.1,
                                   antialiased=True, cmap='jet',
                                   rstride=1, cstride=1)
            plt.colorbar(surf, extend='both')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Time (s)')
            ax.set_zlabel('Amplitude')
        elif style == 'wireframe':
            ax = plt.subplot(111, projection='3d')
            ax.plot_wireframe(time_x, time_y, kernel, linewidth=0.1,
                              antialiased=True, cmap='jet')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Time (s)')
            ax.set_zlabel('Amplitude')

    else:
        print('No plot possible, the kernel is of order {}.'.format(order))


def plot_kernel_freq(vec, kernel, style='surface', title=None, N=20,
                     db=True, unwrap_angle=False, logscale=10):
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

    if order ==1:
        if not title:
            title = 'Transfer kernel of order 1 (linear filter)'
        plt.figure(title)
        plt.clf()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        if logscale:
            ax1.semilogx(vec, kernel_amp, basex=logscale)
            ax2.semilogx(vec, kernel_phase, basex=logscale)
        else:
            ax1.plot(vec, np.abs(kernel))
            ax2.plot(vec, np.angle(kernel))
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(amplabel)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Phase (radians)')

    elif order ==2:
        if not title:
            title = 'Trnsfer kernel of order 2'
        time_x, time_y = np.meshgrid(vec, vec)
        plt.figure(title)
        plt.clf()

        if style == 'contour':
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
            ax1.contourf(time_x, time_y, kernel_amp, N)
            ax2.contourf(time_x, time_y, kernel_phase, N)
        if style == 'surface':
            ax1 = plt.subplot(211, projection='3d')
            ax2 = plt.subplot(212, projection='3d')
            ax1.plot_surface(time_x, time_y, kernel_amp,
                                     linewidth=0.1, antialiased=True,
                                     cmap='jet', rstride=1, cstride=1)
            ax2.plot_surface(time_x, time_y, kernel_phase,
                                     linewidth=0.1, antialiased=True,
                                     cmap='jet', rstride=1, cstride=1)
        if style == 'wireframe':
            ax1 = plt.subplot(211, projection='3d')
            ax2 = plt.subplot(212, projection='3d')
            ax1.plot_wireframe(time_x, time_y, kernel_amp, linewidth=0.1,
                             antialiased=True, cmap='jet')
            ax2.plot_wireframe(time_x, time_y, kernel_phase, linewidth=0.1,
                             antialiased=True, cmap='jet')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Time (s)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Time (s)')
        if not (style == 'contour'):
            ax1.set_zlabel(amplabel)
            ax2.set_zlabel('Phase (radians)')

    else:
        print('No plot possible, the kernel is of order {}.'.format(order))


#==============================================================================
# Main script
#==============================================================================

if __name__ == '__main__':
    """
    Main script for testing.
    """

    import numpy as np

    time_vec = np.arange(5, step=1/100)
    sig_1 = np.sin(2 * np.pi * time_vec)
    sig_2 = np.minimum(sig_1, 0.8)
    sig_3 = np.exp(2j * np.pi * time_vec)
    sig_4 = np.exp(2j * 1.5 * np.pi * time_vec)

    plot_sig_io(sig_1, sig_2, time_vec, name='Test réel', ylim=[-1.1, 1.1])
    plot_sig_io(sig_3, sig_4, time_vec, name='Test complexe', xlim=[0, 3])

    plot_sig_coll(np.stack((sig_1, sig_2, sig_1 - sig_2), axis=1),
                  time_vec, name='Test réel (Collection)', ylim=[-1.1, 1.1],
                  title_plots=['Sinus', 'Cosinus', 'Sinus saturated'])
    plot_sig_coll(np.stack((sig_3, sig_4), axis=1), time_vec, xlim=[0, 3],
                  name='Test complexe (Collection)')
