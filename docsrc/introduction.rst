.. include:: README.rst

For developers
==============


Building docs
-------------
Here's a short guide on how to build the docs on Ubuntu/Debian.

Before building docs, make sure you have adequate dependencies installed:
::

    pip3 install rst-to-myst==0.3.3 sphinx==4.0 -U

It is also necessary for ``rcognita``'s dependencies to be installed,
which can be accomplished by running the following inside ``rcognita``'s repository folder:
::

   python3 setup.py install .

Once the dependencies are installed proceed to execute the following in ``rcognita``'s repostitory
folder:
::

    cd docsrc
    make


Note that the contents of ``README.rst`` are automatically incorporated into the docs.


Contributing
------------

If you'd like to contribute, please contact Pavel Osinenko
via `p.osinenko@gmail.com <mailto:p.osinenko@gmail.com>`__ .

If you'd like to request features or report bugs, please post respective issues
to the `repository <https://gitflic.ru/project/aidynamicaction/rcognita/issue?status=OPEN>`__ .



Forking
-------

When forking rcognita, please, be sure to either delete the docs or modify them in such a way that it
becomes clear that your fork is not in fact the original ``rcognita``.
