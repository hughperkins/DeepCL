# DeepCL Python wrappers

Python wrapper for  [DeepCL](https://github.com/hughperkins/DeepCL)

## Pre-requisites

You must have first installed and activated DeepCL native libraries, see [Build.md](https://github.com/hughperkins/DeepCL/blob/8.x/doc/Build.md)

## To install from pip

```bash
pip install --pre --upgrade DeepCL
```

* related pypi page: [https://pypi.python.org/pypi/DeepCL](https://pypi.python.org/pypi/DeepCL)

## How to use

See [test_deepcl.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_deepcl.py) for an example of:

* creating a network, with several layers
* loading mnist data
* training the network using a higher-level interface (`NetLearner`)

The same example, using numpy arrays: [test_deepcl_numpy.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_deepcl_numpy.py)

For examples of using lower-level entrypoints, see [test_lowlevel.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_lowlevel.py):

* creating layers directly
* running epochs and forward/backprop directly
* note that you need `numpy` installed to run this example

For example of using q-learning, see [test_qlearning.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_qlearning.py).

## To install from source

### Pre-requisites:

* on Windows:
  * Python 2.7 or Python 3.4
  * A compiler:
    * Python 2.7 build: need [Visual Studio 2008 for Python 2.7](http://www.microsoft.com/en-us/download/details.aspx?id=44266) from Microsoft
    * Python 3.4 build: need Visual Studio 2010, eg [Visual C++ 2010 Express](https://www.visualstudio.com/downloads/download-visual-studio-vs#DownloadFamilies_4)
* on linux:
  * Python 2.7 or Python 3.4
  * g++, supporting c++0x, eg 4.4 or higher
* have first already built the native libraries, see [Build.md](../doc/Build.md)
* have activated the native library installation, ie called `dist/bin/activate.sh`, or `dist/bin/activate.bat`

### To install:

```bash

cd python
python setup.py install
```

## Changes

* 25 July 2016:
  * added RandomSingleton class, to set the seed for weights initialization
  * added [xor.py](examples/xor.py) example

