#!/bin/bash

# this is just for prototyping, so not terribly clean for now
# Assumptions :
# - Ubuntu 14.04 linux
# - libDeepCL.so built into ../build
# - 4-core machine (hence  '-j 4', but you can modify this of course)
# - following are on the PATH (eg using `sudo apt-get install`):
#    - luajit
#    - liblua5.1.0-dev
#    - swig
# - mnist data directory is at ../data/mnist (eg, via symlink)

if [[ x${NO_BUILD} == x ]]; then {
    (cd ../build; make -j 4)  # this builds the .so's into ../build
} fi

# echo running cog...
# cog.py -r --verbosity=1 LuaWrappers.h LuaWrappers.lua || exit 1

echo running swig...
swig -c++ -lua -I../src -I/usr/include/lua5.1/ DeepCL.i || exit 1

# should migrate g++ call to CMake sooner or later
echo running g++...
g++ -g -shared -I../src -I../OpenCLHelper -I../qlearning -std=c++0x -I/usr/include/lua5.1/ -o luaDeepCL.so -fPIC DeepCL_wrap.cxx -L../build -lDeepCL || exit 1

echo 'build finished successfully'

# echo running luajit:
# LD_LIBRARY_PATH=.:../build luajit test_lua.lua $1 $2 || exit 1


