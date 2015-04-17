#!/bin/bash

# dev-build.sh: runs swig, which you want to do if you're
# actually maintaining DeepCL itself
# 
# otherwise you can just use build.sh, which doesnt need swig
#
# this is just for prototyping, so not terribly clean for now
# Assumptions :
# - Ubuntu 14.04 linux
# - libDeepCL.so built into ../build
# - 4-core machine (hence  '-j 4', but you can modify this of course)
# - following are on the PATH (eg using `sudo apt-get install`):
#    - swig

if [[ x${NO_BUILD} == x ]]; then {
    (cd ../build; make -j 4)  # this builds the .so's into ../build
} fi

echo running swig...
swig -c++ -lua -I../src -Ithirdparty/lua5.1 DeepCL.i || exit 1

# should migrate g++ call to CMake sooner or later
echo running g++...
g++ -g -shared -I../src -I../OpenCLHelper -I../qlearning -std=c++0x -Ithirdparty/lua5.1 -o luaDeepCL.so -fPIC DeepCL_wrap.cxx -L../build -lDeepCL || exit 1

echo 'build finished successfully'


