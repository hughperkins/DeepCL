# Lua Wrappers

## Concept

Lua wrappers are available.  [luajit](http://luajit.org) is becoming big in machine-learning.

## Demo

* For a demo of high-level functions, to create a network and train it, you can have a look at the method `test_basic` in [test_lua.lua](test_lua.lua)
* For a demo of constructing layers by hand, and handling low-level propagating, batch by batch, you can look at the method `test_lowlevel`, in the same module, ie in [test_lua.lua](test_lua.lua).
* For a demo of q-learning, you can look at [test_qlearning.lua](test_qlearning.lua)

## To build, linux

### Pre-requisites

* cmake
* g++
* swig
* lua development libraries (eg `sudo apt-get install liblua5.1-0-dev`) 
* An OpenCL-compatible driver installed, and OpenCL-compatible GPU

### Procedure

From this directory:
```bash
mkdir build
cd build
cmake ..
make -j 4
```

## To build, Windows (untested)

### Pre-requisites

* cmake
* Visual Studio Express 2010, or later
* swig (should be in the PATH)
* lua development libraries
* An OpenCL-compatible driver installed, and OpenCL-compatible GPU

### Procedure

- open cmake, point at the `lua` directory, and set to build in the `lua\build` subdirectory
  - accept `yes` to create the new directory
  - click `configure`
  - select appropriate generator, eg Visual Studio 2010, according to which one you have
  - click `generate`
- open visual studio, and load any of the projects in the `build` directory
  - change release type to `Release`
  - choose `build` from the `build` menu

## To run

### On linux

According to which module you want to run, you can run one of:
```bash
./run.sh test_lua.lua
```
or:
```bash
./run.sh test_qlearning.lua
```

### On Windows

(this section looking for authors :-) )

## Unit-testing

* The source-code includes the thirdparty lua unit-test tool luaunit.  [test_lua.lua](test_lua.lua)
creates some first initial unit tests

