# DeepCL Python wrappers

Python wrapper for  [DeepCL](https://github.com/hughperkins/DeepCL)

# Pre-requisites

You must have first installed and activated DeepCL native libraries, see [Build.md](https://github.com/hughperkins/DeepCL/blob/8.x/doc/Build.md)

# To install from pip (note, 8.x not in pip yet...)

```bash
pip install DeepCL 
```

* related pypi page: [https://pypi.python.org/pypi/DeepCL](https://pypi.python.org/pypi/DeepCL)

# How to use

See [test_deepcl.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_deepcl.py) for an example of:

* creating a network, with several layers
* loading mnist data
* training the network using a higher-level interface (`NetLearner`)

For examples of using lower-level entrypoints, see [test_lowlevel.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_lowlevel.py):

* creating layers directly
* running epochs and forward/backprop directly

For example of using q-learning, see [test_qlearning.py](https://github.com/hughperkins/DeepCL/blob/master/python/test_qlearning.py).

# To install from source

## Pre-requisites:

* on Windows:
  * Python 2.7 or Python 3.4
  * A compiler:
    * Python 2.7 build: need [Visual Studio 2008 for Python 2.7](http://www.microsoft.com/en-us/download/details.aspx?id=44266) from Microsoft
    * Python 3.4 build: need Visual Studio 2010, eg [Visual C++ 2010 Express](https://www.visualstudio.com/downloads/download-visual-studio-vs#DownloadFamilies_4)
* on linux:
  * Python 2.7 or Python 3.4
  * g++, supporting c++0x, eg 4.4 or higher

## To install:

```bash
cd python
python setup.py install
```

## Notes on how the wrapper works

* [cDeepCL.pxd](https://github.com/hughperkins/DeepCL/blob/master/python/cDeepCL.pxd) contains the definitions of the underlying DeepCL c++ libraries classes
* [PyDeepCL.pyx](https://github.com/hughperkins/DeepCL/blob/master/python/PyDeepCL.pyx) contains Cython wrapper classes around the underlying c++ classes
* [setup.py](https://github.com/hughperkins/DeepCL/blob/master/python/setup.py) is a setup file for compiling the `PyDeepCL.pyx` Cython file

## Maintainer/development information

If you want to modify the python wrappers, you'll need to re-run Cython.  This is no longer handled by `setup.py`, but is handled by the cmake build.  So, to run cython you'll need to:
- install Cython, eg `pip install cython`
- follow the instructions for the native build, [Build.md](https://github.com/hughperkins/DeepCL/blob/8.x/doc/Build.md)
- when you open `ccmake`:
  - enable option `Maintainer options`, then press `c`/`configure`
  - enable `BUILD_PYTHON_WRAPPERS`, then `c`/`configure`
  - enable `DEV_RUN_CYTHON`, then `c`/`configure`
- => and now `g`/`generate`, and build

* If you want to update this readme, you need to re-generate the README.rst, so you'll need pypandoc:
```
pip install pypandoc
```
  * (note that pypandoc depends on pandoc native library)

And then to regenerate README.rst:
```
python setup.py sdist
```


