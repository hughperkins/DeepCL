DeepCL Python wrappers
======================

Python wrapper for `DeepCL <https://github.com/hughperkins/DeepCL>`__

Pre-requisites
--------------

-  You must have first installed and activated DeepCL native libraries,
   see
   `Build.md <https://github.com/hughperkins/DeepCL/blob/8.x/doc/Build.md>`__
-  ``numpy``

To install from pip
-------------------

.. code:: bash

    pip install --upgrade DeepCL

-  related pypi page: https://pypi.python.org/pypi/DeepCL

How to use
----------

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

To install from source
----------------------

Pre-requisites:
~~~~~~~~~~~~~~~

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
-  have first already built the native libraries, see
   `Build.md <../doc/Build.md>`__
-  have activated the native library installation, ie called
   ``dist/bin/activate.sh``, or ``dist/bin/activate.bat``
-  ``numpy`` installed

To install:
~~~~~~~~~~~

.. code:: bash


    cd python
    python setup.py install

Changes
-------

-  30 July 2016:
-  Added ``net.getNetdef()``. Note that this is only an approximate
   representation of the network
-  29 July 2016:
-  New feature: can provide image tensor as 4d tensor now ,instead of 1d
   tensor (1d tensor ok too)
-  CHANGE: all image and label tensors must be provided as numpy tensors
   now, ``array.array`` no longer valid input
-  bug fix: qlearning works again :-)
-  25 July 2016:
-  added RandomSingleton class, to set the seed for weights
   initialization
-  added `xor.py <examples/xor.py>`__ example
