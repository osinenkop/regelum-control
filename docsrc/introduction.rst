.. include:: README.rst

Getting started
===============

The package is organized in modules.

These are:

* ``controllers``

* ``loggers``

* ``models``

* ``simulator``

* ``systems``

* ``utilities``

* ``visuals`` 

There is a collection of main modules (presets) for each agent-environment configuration.

To work with ``rcognita``, use one of the presets by ``python`` running it and specifying parameters.
If you want to create your own environment, fork the repo and implement one in ``systems`` via inheriting the ``System`` superclass.

For developers
==============

In Linux-based OS, to build these wiki docs, run inside cloned repo folder:

::

    cd docsrc
    make

Before running make, you need to make sure that all dependencies are installed and sphinx your sphinx is fresh (i.e somewhere around 5.3.0). On the website (of sphinx) it tells you to use `apt install python3-sphinx`, but THIS IS A TRAP. If you do this you'll just install an outdated version of sphinx. You probably want to `pip install -U sphinx` instead. Also be sure to also install the theme via  `pip install sphinx_rtd_theme`.
