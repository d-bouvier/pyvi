# pyVI - Python Toolbox for Volterra System Identification
This project proposes a python toolbox for Nonlinear System Identification, developped by Damien Bouvier during his PhD.


================
= GIT Workflow =
================

Fixed Branches
==============
- master
	Working state of the algorithms
	(to be tagged with version number: '#version_desc')
	branch off <- /
	merge into -> publish
- dev
	Development state of the algorithms
	branch off <- master
	merge into -> master, publish
- publish [publish/nameArticle]
	Version for article and/or presentations, with script for the figure creation
	branch off <- master
	merge into -> /

Temporary Branches
==================
- feature [feature/nameFeature]
	Development state of the algorithm for a specific idea/task
	branch off <- dev
	merge into -> dev
- debug [debug/nameBug]
	Debug branch for errors that need to be resolved immediately
	branch off <- master
	merge into -> dev, master