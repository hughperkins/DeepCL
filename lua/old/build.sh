#!/bin/bash

# just does a build
# if you want to actually modify the sourcecode, you probably
# want `dev-build.sh`, which will also run swig
# this is just for prototyping, so not terribly clean for now
# Assumptions :
# - Ubuntu 14.04 linux
# - libDeepCL.so built into ../build

if [[ x${NO_BUILD} == x ]]; then {
    (cd ../build; make )  # this builds the .so's into ../build
} fi

# should migrate g++ call to CMake sooner or later
echo running g++...
g++ -g -shared -I../src -I../OpenCLHelper -I../qlearning -std=c++0x -Ithirdparty/lua5.1 -o luaDeepCL.so -fPIC DeepCL_wrap.cxx -L../build -lDeepCL || exit 1

echo 'build finished successfully'



