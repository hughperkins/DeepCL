PyClConvolve
============

Python wrapper for
`ClConvolve <https://github.com/hughperkins/ClConvolve>`__

How to use
==========

See `test\_clconvolve.py <PyClConvolve/test_clconvolve.py>`__ for an
example of:

-  creating a network, with several layers
-  loading mnist data
-  training the network using a higher-level interface (``NetLearner``)

For examples of using lower-level entrypoints, see
`test\_lowlevel.py <https://github.com/hughperkins/PyClConvolve/blob/master/test_lowlevel.py>`__:

-  creating layers directly
-  running epochs and forward/backprop directly

Notes on how the wrapper works
==============================

-  `cClConvolve.pxd <https://github.com/hughperkins/ClConvolve/blob/master/PyClConvolve/cClConvolve.pxd>`__
   contains the definitions of the underlying ClConvolve c++ libraries
   classes
-  `PyClConvolve.pyx <https://github.com/hughperkins/ClConvolve/blob/master/PyClConvolve/PyClConvolve.pyx>`__
   contains Cython wrapper classes around the underlying c++ classes
-  `setup.py <https://github.com/hughperkins/ClConvolve/blob/master/PyClConvolve/setup.py>`__
   is a setup file for compiling the ``PyClConvolve.pyx`` Cython file

To install from pip
===================

.. code:: bash

    pip install PyClConvolve 

-  related pypi page: https://pypi.python.org/pypi/PyClConvolve

To build directly
=================

Pre-requisites:
---------------

Compillers
~~~~~~~~~~

-  on Windows:
-  Python 2.7 build: need `Visual Studio 2008 for Python
   2.7 <http://www.microsoft.com/en-us/download/details.aspx?id=44266>`__
   from Microsoft
-  Python 3.4 build: need Visual Studio 2010, eg `Visual C++ 2010
   Express <https://www.visualstudio.com/downloads/download-visual-studio-vs#DownloadFamilies_4>`__

Python packages
~~~~~~~~~~~~~~~

-  Need the following python packages installed, eg via ``pip install``:
-  cython

To build:
---------

.. code:: bash

    cd PyClConvolve
    python setup.py build_ext -i

Considerations for Python wrapper developers
============================================

-  By default, cython disables ctrl-c, so need to handle this ourselves
   somehow, eg see ``NetLearner.learn`` for an example
-  To handle ctrl-c, we need to use ``nogil``, which means we cant use
   the ``except +`` syntax, I think, hence need to handle this ourselves
   too :-) see again \`Netlearner.learn for an example

To build, obsolete method
=========================

Should probably more or less build on Windows too, but here are
instructions for linux for now:

-  checkout:

   ::

       git clone --recursive https://github.com/hughperkins/PyClConvolve.git

-  build the C++ library:

   ::

       cd ClConvolve
       mkdir build
       cd build
       cmake ..
       make -j 4
       cd ../..

-  build the Cython modules

   ::

       CFLAGS="-IClConvolve/src -IClConvolve/OpenCLHelper -std=c++11" LDFLAGS="-LClConvolve/build" python setup.py build_ext -i 2>&1 | less

-  run one of the example scripts:

   ::

       LD_LIBRARY_PATH=ClConvolve/build python test_clconvolve.py
       LD_LIBRARY_PATH=ClConvolve/build python test_lowlevel.py


