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
- g++ (should support c++0x; eg 4.4 or better)
- An OpenCL-compatible driver installed, and OpenCL-compatible GPU

*Optional:*
- libjpeg62 or compatible, eg `sudo apt-get install libjpeg-turbo8-dev` (libjpeg-turbo is faster than original libjpeg6.2, by around 2-4 times, because it uses SIMD extensions)

### Procedure

```bash
git clone --recursive https://github.com/hughperkins/DeepCL.git
cd DeepCL
# if you need to use a specific branch, then choose that now, ie `git checkout some-branch-name`
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
- Visual Studio (current 'standard' build system is: Visual Studio 2010 Express, but should also work on Visual Studio 2008 for Python 2.7, and Visual Studio Express 2013)
- An OpenCL-compatible driver installed, and OpenCL-compatible GPU

*Optional:*
- (new) libjpeg62, or compatible, eg [libjpeg-turbo](http://www.libjpeg-turbo.org/Documentation/OfficialBinaries)  (libjpeg-turbo is faster than original libjpeg6.2, by around 2-4 times, because it uses SIMD extensions)
  - if you want, I made a fresh build of libjpeg-turbo 1.4.0:
    - dynamic library (doesnt work for me): [libjpeg-turbo-1.4.0-win32.zip](http://deepcl.hughperkins.com/Downloads/turbojpeg-1.4.0-win32.zip) and [libjpeg-turbo-1.4.0-win64.zip](http://deepcl.hughperkins.com/Downloads/turbojpeg-1.4.0-win64.zip)
    - static library (works ok for me): [libjpeg-turbo-1.4.0-win32.zip](http://deepcl.hughperkins.com/Downloads/turbojpeg-1.4.0-win32-static.zip) and [libjpeg-turbo-1.4.0-win64.zip](http://deepcl.hughperkins.com/Downloads/turbojpeg-1.4.0-win64-static.zip)
- Python 2.7 or Python 3.4 (needs python, and also the development library and include files)

### Procedure

- in git, do `git clone --recursive https://github.com/hughperkins/DeepCL.git`
  - note: recently, moved EasyCL/thirdparty/clew from submodule to simply copying in the files
  - hopefully this makes new clones easier, but for now, if you already have a clone, when you next update, you might need to first remove the EasyCL/thirdparty/clew directory
- if you need to use a specific branch, then choose that now, ie `git checkout some-branch-name`
- create a subdirectory `build` in the git cloned `DeepCL` directory
- open cmake, point at the `DeepCL` directory, and set to build in the `build` subdirectory
  - `configure`, select 'visual studio 2010' (or as appropriate)
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

