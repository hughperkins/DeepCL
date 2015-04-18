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

### Procedure

* create a subdirectory 'build'
* open cmake, and set source to this directory, containing this README.md, and build directory to the 'build' subdirectory
* 'configure'
* 'generate'
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

## Why two python directories

* the `python` directory uses Cython to create the wrappers
* this `python_swig` directory uses swig

## What are the future of these two directories?

* Since using swig is significantly lower maintenance, because portable languages, eg between python and lua,
the swig wrappers will almost certainly replace the cython wrappers, and the cython wrappers will become
deprecated

