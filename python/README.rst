DeepCL Python wrappers
======================

Python wrapper for `DeepCL <https://github.com/hughperkins/DeepCL>`__

To install from pip
===================

.. code:: bash

    pip install DeepCL 

-  related pypi page: https://pypi.python.org/pypi/DeepCL

How to use
==========

See
`test\_deepcl.py <https://github.com/hughperkins/DeepCL/blob/master/python/test_deepcl.py>`__
for an example of:

-  creating a network, with several layers
-  loading mnist data
-  training the network using a higher-level interface (``NetLearner``)

For examples of using lower-level entrypoints, see
`test\_lowlevel.py <https://github.com/hughperkins/DeepCL/blob/master/python/test_lowlevel.py>`__:

-  creating layers directly
-  running epochs and forward/backprop directly

For example of using q-learning, see
`test\_qlearning.py <https://github.com/hughperkins/DeepCL/blob/master/python/test_qlearning.py>`__.

To build from source
====================

Pre-requisites:
---------------

-  on Windows:
-  Python 2.7 or Python 3.4
-  A compiler:

   -  Python 2.7 build: need `Visual Studio 2008 for Python
      2.7 <http://www.microsoft.com/en-us/download/details.aspx?id=44266>`__
      from Microsoft
   -  Python 3.4 build: need Visual Studio 2010, eg `Visual C++ 2010
      Express <https://www.visualstudio.com/downloads/download-visual-studio-vs#DownloadFamilies_4>`__

-  on linux:
-  Python 2.7 or Python 3.4
-  g++, supporting c++0x, eg 4.4 or higher

To build:
---------

.. code:: bash

    cd python
    python setup.py build_ext -i

Then, you can run from this directory, by making sure to add it to the
path, eg:

::

    PYTHONPATH=. python test_lowlevel.py /my/mnist/data/dir 

To install:
-----------

.. code:: bash

    cd python
    python setup.py install

Notes on how the wrapper works
------------------------------

-  `cDeepCL.pxd <https://github.com/hughperkins/DeepCL/blob/master/python/cDeepCL.pxd>`__
   contains the definitions of the underlying DeepCL c++ libraries
   classes
-  `PyDeepCL.pyx <https://github.com/hughperkins/DeepCL/blob/master/python/PyDeepCL.pyx>`__
   contains Cython wrapper classes around the underlying c++ classes
-  `setup.py <https://github.com/hughperkins/DeepCL/blob/master/python/setup.py>`__
   is a setup file for compiling the ``PyDeepCL.pyx`` Cython file

to run unit-tests
-----------------

From the python directory:

.. code:: bash

    nosetests -sv

Development builds
------------------

-  If you want to modify the sourcecode, you'll need to re-run cython,
   so you'll need cython:

   ::

       pip install cython

-  If you want to update this readme, you might want to re-generate the
   README.rst, so you'll need pypandoc:

   ::

       pip install pypandoc

-  (note that pypandoc depends on pandoc)


