# PyDeepCL

Python wrapper for  [DeepCL](https://github.com/hughperkins/DeepCL)

# How to use

See [test_clconvolve.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_clconvolve.py) for an example of:

* creating a network, with several layers
* loading mnist data
* training the network using a higher-level interface (`NetLearner`)

For examples of using lower-level entrypoints, see [test_lowlevel.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_lowlevel.py):

* creating layers directly
* running epochs and forward/backprop directly

# Notes on how the wrapper works

* [cDeepCL.pxd](https://github.com/hughperkins/DeepCL/blob/master/python/cDeepCL.pxd) contains the definitions of the underlying DeepCL c++ libraries classes
* [PyDeepCL.pyx](https://github.com/hughperkins/DeepCL/blob/master/python/PyDeepCL.pyx) contains Cython wrapper classes around the underlying c++ classes
* [setup.py](https://github.com/hughperkins/DeepCL/blob/master/python/setup.py) is a setup file for compiling the `PyDeepCL.pyx` Cython file

# To install from pip

```bash
pip install DeepCL 
```

* related pypi page: [https://pypi.python.org/pypi/DeepCL](https://pypi.python.org/pypi/DeepCL)

# To build directly

## Pre-requisites:

### Compilers
* on Windows:
  * Python 2.7 build: need [Visual Studio 2008 for Python 2.7](http://www.microsoft.com/en-us/download/details.aspx?id=44266) from Microsoft
  * Python 3.4 build: need Visual Studio 2010, eg [Visual C++ 2010 Express](https://www.visualstudio.com/downloads/download-visual-studio-vs#DownloadFamilies_4)

### Python packages

* Need the following python packages installed, eg via `pip install`:
  * cython

## To build:

```bash
cd python
python setup.py build_ext -i
```

# Considerations for Python wrapper developers

* By default, cython disables ctrl-c, so need to handle this ourselves somehow, eg see `NetLearner.learn` for an example
* To handle ctrl-c, we need to use `nogil`, which means we cant use the `except +` syntax, I think, hence need to handle this ourselves too :-)  see again `Netlearner.learn for an example

## to run unit-tests

(Draft) From this directory:

```bash
nosetests -sv
```

