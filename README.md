# pyvi - Python Toolbox for Volterra System Identification
This project proposes a python toolbox for Volterra Series Identification using nonlinear homogeneous order separation.

, developped by Damien Bouvier during his PhD.

A Python software (Py) dedicated to the simulation of multi-physical Port-Hamiltonian Systems (PHS) described by graph structures.

The PHS formalism decomposes network systems into **conservative** parts, **dissipative** parts and **source** parts, which are combined according to an **energy conserving interconnection**. This approach permits to formulate the dynamics of multi-physical systems as a **set of differential-algebraic equations** structured according to energy flows. This **structure** proves the **passivity** of the system, including the nonlinear cases. Moreover, it guarantees the **stability** of the numerical simulations for an adapted structure preserving numerical method.

License
=======
`Pyvi <https://github.com/d-bouvier/pyvi/>`_ is distributed under the BSD 3-Clause "New" or "Revised" License.

Python prerequisites
====================
The `Pyvi <https://github.com/d-bouvier/pyvi/>`_ package needs the following packages installed:

- `numpy <http://www.numpy.org>`_
- `scipy <http://www.scipy.org>`_

The package has been fully tested with the following versions:

- python 3.6.4
- numpy 1.14.0
- scipy 0.19.1

Package structure
=================

The package is divided into the following folders:

* [/pyvi/separation](https://github.com/d-bouvier/pyvi/tree/master/pyvi/separation)
Module for nonlinear homogeneous order separation of Volterra series.

* [/pyvi/identification](https://github.com/d-bouvier/pyvi/tree/master/pyvi/identification)
Module for Volterra kernels identification.

* [/pyvi/volterra](https://github.com/d-bouvier/pyvi/tree/master/pyvi/volterra)
Module creating various tools for Volterra series.

* [/pyvi/utilities](https://github.com/d-bouvier/pyvi/tree/master/pyvi/utilities)
Module containing various useful class and functions.
