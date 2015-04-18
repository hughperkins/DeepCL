# Python wrappers, using SWIG

## Concept

This directory contains python wrappers built using swig

## To build

### Pre-requisites

* swig
* cmake
* compiler, ie:
  * visual studio 2010, on Windows, or
  * g++, supporting c++0x, on linux
* python development libraries (python.h, etc)

### On linux

* From this directory, the one containing this README.md:
```bash
mkdir -p build
cd build
cmake ..
make -j 4
cd ..
```

### On Windows (untested)

* create a subdirectory 'build'
* open cmake, and set source to this directory, containing this README.md, and build directory to the 'build' subdirectory
* press 'configure' button
* press 'generate' button
* on linux:
```bash
cd build
make -j 4
```
* on Windows:
  * open visual studio
  * load one of the project files from the 'build' directory
  * set configuration to 'Release'
  * click on 'build solution'

## To run

### On linux

From this directory, the one with this README.md in, eg:
```bash
PYTHONPATH=build python test_lowlevel.py /norep/data/mnist
```
* You need to change `/norep/data/mnist`, to point to your mnist data directory

### On Windows (untested)

From this directory, the one with this README.md in:
* set PYTHONPATH to the directory where _PyDeepCL.dll was built to
* Run:
```cmd
python test_lowlevel.py c:\data\mnist
```
  * Make sure to change `c:\data\mnist` to the directory path of your mnist data directory
 
## Why two python directories

* the `python` directory uses Cython to create the wrappers
* this `python_swig` directory uses swig

## What are the future of these two directories?

* Since using swig is significantly lower maintenance, because portable languages, eg between python and lua,
the swig wrappers will almost certainly replace the cython wrappers, and the cython wrappers will become
deprecated

