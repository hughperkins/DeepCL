#!/bin/bash

# this is just for prototyping, so not terribly clean for now
# Expectations:
# - Ubuntu 14.04 linux
# - libDeepCL.so should be built into ../build
# - 4-core machine (hence  '-j 4', but you can modify this of course)
# - luajit built source at ~/lua (eg via a symlink is ok)
# - swig is in the PATH (installed via `sudo apt-get install swig`)
# - mnist data directory is at ../data/mnist (eg, via symlink)

#alias luajit='~/lua/src/luajit' # could also add to PATH of course
                            # but this way keeps PATH clean
# (cd ../build; make -j 4)  # this builds the .so's into ../build
swig -c++ -lua -I../src -I$HOME/lua/src DeepCL.i || { echo swig failed; exit 1; }
# should migrate g++ call to CMake sooner or later
g++ -shared -I../src -I../OpenCLHelper -std=c++0x -I$HOME/lua/src -o luaDeepCL.so -fPIC DeepCL_wrap.cxx -L../build -lDeepCL || { echo g++ failed; exit 1; }
#alias
LD_LIBRARY_PATH=.:../build ~/lua/src/luajit test_lua.lua || exit 1

