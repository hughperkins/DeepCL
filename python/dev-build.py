#!/usr/bin/python
# Copyright Hugh Perkins 2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

# ****************************************************************************
# *                                                                          *
# *  IMPORTANT:                                                              *
# *  This script is for python wrapper development                           *
# *  If you want to build and use the wrapper, you probably want             *
# *  to use: 'setup.py'                                                      *
# *                                                                          *
# ****************************************************************************
#
# This script uses the binary built by the c++ cmake build, which can be built
# using multiple threads etc, so is fast to build during development :-)
# If you just want to build and use the wrapper, you should probably use 
# 'setup.py', which builds slower, but more likely to be reliable and multi
# platform
#
# Bearing in mind these caveats, if you do want to use this script:
#
# - first, build DeepCL shared object (.so or .dll) into the ../build directory
#
# - then, simply run this script as for setup.py, ie:
#
#     python dev-build.py build_ext -i
#
# ... and then you can simply run the test python scripts as before, eg:
#
#     ./test_lowlevel.py /mydata/mnist 
#
# The following command might be useful for running the C++ build, on linux:
#
#     ( cd ..; mkdir -p build; cd build; cmake ..; make -j 4 )
#

import os
import os.path
import sysconfig
import sys
import glob
import platform
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import pypandoc
import cogapp

for arg in sys.argv:
    if arg == 'upload' or arg == 'register' or arg == 'testarg':
        print('This setup is not designed to be uploaded or registered :-)')
        sys.exit(-1)

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def get_so_suffix():
    if sysconfig.get_config_var('SOABI') != None:
        return "." + sysconfig.get_config_var('SOABI')
    return ""

pypandoc.convert('README.md', 'rst', outputfile = 'README.rst' )

cog = cogapp.cogapp.Cog()
cog.callableMain(['','--verbosity=1','-r','CyScenario.h','PyDeepCL.pyx','cDeepCL.pxd'])

# from http://stackoverflow.com/questions/14320220/testing-python-c-libraries-get-build-path
def distutils_dir_name(dname):
    """Returns the name of a distutils build directory"""
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)
 
def lib_build_dir():
    return os.path.join('build', distutils_dir_name('lib'))

compile_options = []
osfamily = platform.uname()[0]
if osfamily == 'Windows':
   compile_options.append('/EHsc')
elif osfamily == 'Linux':
   compile_options.append('-std=c++0x')
   compile_options.append('-g')
else:
   pass
   # put other options etc here if necessary

runtime_library_dirs = []
libraries = []
if osfamily == 'Linux':
    runtime_library_dirs= ['../build']

if osfamily == 'Windows':
    libraries = ['winmm']

libraries.append('DeepCL')

ext_modules = [
    Extension("PyDeepCL",
              sources=["PyDeepCL.pyx", 'CyWrappers.cpp'], 
              include_dirs = ['../src','../OpenCLHelper','../qlearning'],
              libraries= libraries,
              extra_compile_args=compile_options,
#              extra_objects=['cDeepCL.pxd'],
              library_dirs = runtime_library_dirs,
              runtime_library_dirs=runtime_library_dirs,
              language="c++"
    )
]

setup(
  name = 'DeepCL',
  # version = "1.0.2",
  author = "Hugh Perkins",
  author_email = "hughperkins@gmail.com",
  description = 'python wrapper for DeepCL deep convolutional neural network library for OpenCL',
  license = 'MPL',
  url = 'https://github.com/hughperkins/DeepCL',
  long_description = read('README.rst'),
  classifiers = [
    'Development Status :: 4 - Beta',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
  ],
  install_requires = ['Cython>=0.22','cogapp>=2.4','future>=0.14.3'],
  tests_require = ['nose>=1.3.4'],
  scripts = ['test_deepcl.py','test_lowlevel.py'],
 # modules = libraries,
#  libraries = libraries,
  ext_modules = cythonize( ext_modules),
)


