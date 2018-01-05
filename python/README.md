# DeepCL Python wrappers

Python wrapper for  [DeepCL](https://github.com/hughperkins/DeepCL)

## Pre-requisites

* You must have first installed and activated DeepCL native libraries, see [Build.md](https://github.com/hughperkins/DeepCL/blob/8.x/doc/Build.md)
* `numpy`

## To install from pip

```bash
pip install --upgrade DeepCL
```

* related pypi page: [https://pypi.python.org/pypi/DeepCL](https://pypi.python.org/pypi/DeepCL)

## How to use

See [test_deepcl.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_deepcl.py) for an example of:

* creating a network, with several layers
* loading mnist data
* training the network using a higher-level interface (`NetLearner`)

For examples of using lower-level entrypoints, see [test_lowlevel.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_lowlevel.py):

* creating layers directly
* running epochs and forward/backprop directly

For example of using q-learning, see [test_qlearning.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_qlearning.py).

## To install from source

### Pre-requisites:

* on Windows:
  * Python 2.7 or Python 3.5
  * A compiler:
    * Python 2.7 build: need [Visual Studio 2008 for Python 2.7](http://www.microsoft.com/en-us/download/details.aspx?id=44266) from Microsoft
    * Python 3.5 build: need Visual Studio 2015, (https://www.visualstudio.com/downloads)
* on linux:
  * Python 2.7 or Python 3.4/3.5
  * g++, supporting c++11, eg 4.6 or higher
* have first already built the native libraries, see [Build.md](../doc/Build.md)
* have activated the native library installation, ie called `dist/bin/activate.sh`, or `dist/bin/activate.bat`
* `numpy` installed

### To install:

```bash

cd python
python setup.py install
```

## Changes

* 30 July 2016:
  * Added `net.getNetdef()`.  Note that this is only an approximate representation of the network
* 29 July 2016:
  * New feature: can provide image tensor as 4d tensor now ,instead of 1d tensor (1d tensor ok too)
  * CHANGE: all image and label tensors must be provided as numpy tensors now, `array.array` no longer valid input
  * bug fix: qlearning works again :-)
* 25 July 2016:
  * added RandomSingleton class, to set the seed for weights initialization
  * added [xor.py](examples/xor.py) example

