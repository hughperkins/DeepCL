<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [To build](#to-build)
  - [linux](#linux)
    - [Pre-requisites](#pre-requisites)
    - [Procedure](#procedure)
  - [Windows](#windows)
    - [Pre-requisites](#pre-requisites-1)
    - [Procedure](#procedure-1)
  - [Linking](#linking)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

#To build

[![Build Status](https://travis-ci.org/hughperkins/DeepCL.svg?branch=master)](https://travis-ci.org/hughperkins/DeepCL)

## Build options

* If you want to be able to read training/testing data from jpeg files, then please choose `BUILD_JPEG_SUPPORT` = `ON`. You will need to provide turbojpeg library and headers, or compatible.  Otherwise set to `OFF`

## linux

### Pre-requisites

*Required:*
- git
- make 
- cmake
- cmake-curses-gui
- gfortran
- g++ (should support c++11; eg 4.6 or better)
- An OpenCL-compatible driver installed, and OpenCL-compatible GPU

*Optional:*
- libjpeg62 or compatible, eg `sudo apt-get install libjpeg-turbo8-dev` (libjpeg-turbo is faster than original libjpeg6.2, by around 2-4 times, because it uses SIMD extensions)

### Procedure

```bash
git clone --recursive https://github.com/hughperkins/DeepCL.git
cd DeepCL
# if you need to use a specific branch, then choose that now, ie:
#    git checkout some-branch-name
#    git submodule update --recursive
mkdir build
cd build
ccmake ..
# in ccmake:
# - press 'c'/configure
# - choose the options you want
# - press 'c' /configure again
# - press 'g' / generate, then `q` / quit
make -j 4 install
```

The outputs will appear in subdirectories of `../dist`

Note:
* be sure to add `--recursive` when you clone, else when you build it will complain about OpenCLHelper missing (or clew missing)
  * if you do forget, you can experiment with running `git submodule init --recursive`, and then `git submodule update --recursive`
* you might need to play around with commands such as `git submodule update --recursive` occasionally, to pull down new OpenCLHelper updates
* note: recently, moved EasyCL/thirdparty/clew from submodule to simply copying in the files
   * hopefully this makes new clones easier, but for now, if you already have a clone, when you next update, you might need to first remove the EasyCL/thirdparty/clew directory

### To activate, setup environment:

Open a bash prompt, and run:
```
source /path/to/DeepCL/dist/bin/activate.sh
```
(where you need to modify `/path/to/DeepCL` appropriately)

Keep the bash open, and go to the next section

### To check all is working

Unit-tests:
```
deepcl_unittests
```
Most tests should pass, but one or two might fail.  Please do feel free to raise an issue for failing tests, even if they fail intermittently.

Commandline training:
```
deepcl_train numtest=-1 numtrain=10000 datadir=/data/mnist
```
(change path to wherever the mnist data files are downloaded)

## Windows

### Pre-requisites

*Required:*
- git
- cmake
- Visual Studio 2015
- An OpenCL-compatible driver installed, and OpenCL-compatible GPU

*Optional:*
- (new) libjpeg62, or compatible, eg [libjpeg-turbo](http://www.libjpeg-turbo.org/Documentation/OfficialBinaries)  (libjpeg-turbo is faster than original libjpeg6.2, by around 2-4 times, because it uses SIMD extensions)
  - the CI builds use http://deepcl.hughperkins.com/Downloads/jpegturbo-1.5-64.zip and http://deepcl.hughperkins.com/Downloads/jpegturbo-1.5-32.zip
- Python 2.7 or Python 3.5 (note: python 3.4 no longer supported, on Windows)

### Procedure

- in git, do `git clone --recursive https://github.com/hughperkins/DeepCL.git`
  - note: recently, moved EasyCL/thirdparty/clew from submodule to simply copying in the files
  - hopefully this makes new clones easier, but for now, if you already have a clone, when you next update, you might need to first remove the EasyCL/thirdparty/clew directory
- if you need to use a specific branch, then choose that now, ie:
  - `git checkout some-branch-name`, and then
  - `git submodule update --recursive`
- create a subdirectory `build` in the git cloned `DeepCL` directory
- open cmake, point at the `DeepCL` directory, and set to build in the `build` subdirectory
  - `configure`, select 'visual studio 2015' (or as appropriate)
- choose the options you want, eg turn python on/off, jpeg on/off
- click `generate`
- open visual studio, and load any of the projects in the `build` directory
  - change release type to `Release`
  - choose `build` from the `build` menu
- select 'INSTALL' project, right-click and 'Build'

The outputs will appear in the subdirectory 'dist'

### To activate, setup environment:

Open a cmd prompt, and run:
```
call \path\to\DeepCL\dist\bin\activate.bat
```
(where you need to modify `\path\to\DeepCL` appropriately)

Keep the cmd open, and go to the next section

### To check all is working

First open a cmd prompt, and activate, as above, then:

Unit-tests:
```
deepcl_unittests
```
Most tests should pass, but one or two might fail.  Please do feel free to raise an issue for failing tests, even if they fail intermittently.

Commandline training:
```
deepcl_train numtest=-1 numtrain=10000 datadir=c:\data\mnist
```
(change path to wherever the mnist data files are downloaded)

## Linking

If you want to use the DeepCL library from C++, you will need to link with the following libraries:
- libDeepCL.so (or DeepCL.dll, on Windows)
- libEasyCL.so (or EasyCL.dll, on Windows)
- libclew.so / clew.dll
- libclBLAS.so / clBLAS.dll

## Some errors, and possible causes

- during build, `fatal error: CppRuntimeBoundary.h: No such file or directory`
  - make sure you ran `source ../dist/bin/activate.sh`, or similar (see above for exact command similar to this)

# Building libjpeg

Using instructoins at https://stackoverflow.com/questions/12652178/compiling-libjpeg/19045485#19045485
- downloaded http://www.ijg.org/files/jpegsr9b.zip , unzipped
- downloaded http://www.bvbcode.com/code/f2kivdrh-395674-down , copied to unzipped jpeg-9b directory
- open cmd
- run `call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\vcvars32.bat"`
- run `nmake /f makefile.vc setup-v10`
- open `jpeg.sln` in msvc2015, and build
- (dont need `apps.sln`, ignore it)
- you've built win32, into `release` directory
  - copy `jpeg.lib` somewhere
- now, redo, but selecting `x64` in the dropdown, in the menubar, at hte top of msvc ui
  - you have to do something like right-click, 'add configuration', and select architecture 'x64' to do this
  - this will build into `x64\release`, instead of simply `release`

This builds as a static library, so no need to distribute dlls and stuff
