# Lua Wrappers

## Concept

Lua wrappers are available.  [luajit](http://luajit.org) is becoming big in machine-learning.

## Demo

* For a demo of high-level functions, to create a network and train it, you can have a look at the method `test_basic` in [test_deepcl.lua](test_deepcl.lua)
* For a demo of constructing layers by hand, and handling low-level propagating, batch by batch, you can look at the method `test_lowlevel`, in the same module, ie in [test_deepcl.lua](test_deepcl.lua).
* For a demo of q-learning, you can look at [test_qlearning.lua](test_qlearning.lua)

## Luarocks build

### Installation from luarocks

* There is a source rock available on luarocks [luadeepcl](http://luarocks.org/modules/hughperkins/luadeepcl):

```
luarocks install --server=http://luarocks.org luadeepcl
```
* This builds from source, just as for the below, so this does have the same pre-requisites as building 
directly from github source, ie:
  * cmake
  * lua development libraries (eg `sudo apt-get install liblua5.1-0-dev`)
  * a C++ compiler, supporting c++0x
* You'll also need a working OpenCL-enabled platform, eg OpenCL-enabled GPU, or OpenCL-enabled CPU

### To build, using luarocks, linux

#### Prerequisites

* cmake
* g++, supporting c++0x, on linux
* lua development libraries (eg `sudo apt-get install liblua5.1-0-dev`)
* luarocks
* luajit

#### Procedure

From this directory:
```bash
luarocks make
```

### To run, using luarocks

Simple run the test scripts, like:
```bash
luajit test_deepcl.lua
```

## Unit-testing

* The source-code includes the thirdparty lua unit-test tool luaunit.  [test_deepcl.lua](test_deepcl.lua)
creates some first initial unit tests

## To build a rock

To build a source rock, use linux:
* first set the version in version.txt to your desired version
* then run:
```
./pack.sh
```
* whilst `pack.sh` only runs on linux, hopefully the resulting rock should be cross-platform.  Hopefully.  Let
me know any issues please :-)

## Development builds

For development, we need swig installed:
* [swig](http://www.swig.org)

And we need to build using cmake.  On linux, this looks something like:
```
mkdir build
cd build
ccmake ..
# click 'c' for configure
# turn on the option 'RUN_SWIG'
# click 'c' again
# then 'g' to generate
make -j 4
```

To run development builds, you can simply use the luarocks method above.  Alternatively, you can also try to
use the library you built using cmake just now, using run.sh, eg:
```
./run.sh lua_deepcl.lua
```

