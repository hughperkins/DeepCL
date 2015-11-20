# Copyright Hugh Perkins 2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import print_function
import os
import os.path
import sys
import platform
from setuptools import setup
from setuptools import Extension

cython_present = False

building_dist = False
for arg in sys.argv:
    if arg in ('sdist', 'bdist', 'bdist_egg', 'build_ext'):
        building_dist = True
        break

if building_dist:
    try:
        import pypandoc
        pypandoc.convert('README.md', 'rst', outputfile='README.rst')
    except:
        print('WARNING: pypandoc not installed, cannot update README.rst')


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

compile_options = []
osfamily = platform.uname()[0]
if osfamily == 'Windows':
    compile_options.append('/EHsc')
elif osfamily == 'Linux':
    compile_options.append('-std=c++0x')
    compile_options.append('-g')
elif osfamily == 'Darwin':
    compile_options.append('-mmacosx-version-min=10.7')
    compile_options.append('-stdlib=libc++')
    compile_options.append('-g')
else:
    print('WARNING: your osfamily "{os}" not recognized.'.format(
        os=osfamily))
    print('Please raise an issue at https://github.com/hughperkins/DeepCL/issues/new')
    pass
    # put other options etc here if necessary

compile_options.append('-DUSE_CLEW')

runtime_library_dirs = []
libraries = []
libraries.append('clBLAS')
libraries.append('EasyCL')
libraries.append('DeepCL')

library_dirs = []
library_dirs.append('../dist/lib')
library_dirs.append('../dist/lib/import')

if osfamily == 'Linux':
    runtime_library_dirs = ['.']

if osfamily == 'Windows':
    libraries.append('winmm')

sources = ["PyDeepCL.cxx", 'CyWrappers.cpp']
if cython_present:
    sources = ["PyDeepCL.pyx", 'CyWrappers.cpp']
ext_modules = [
    Extension("PyDeepCL",
              sources=sources,
              include_dirs=['mysrc', 'mysrc/lua'],
              library_dirs=library_dirs,
              libraries=libraries,
              extra_compile_args=compile_options,
              runtime_library_dirs=runtime_library_dirs,
              language="c++")]


def read_if_exists(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.isfile(filepath):
        return open(filepath).read()
    else:
        return ""

version = read_if_exists('version.txt').strip().replace('v', '')
if building_dist and version == '':
    raise Exception('version cannot be empty string when building dist')
print('version: ', version)

setup(
    name='DeepCL',
    version=version,
    author="Hugh Perkins",
    author_email="hughperkins@gmail.com",
    description=(
        'python wrapper for DeepCL deep convolutional '
        'neural network library for OpenCL'),
    license='MPL',
    url='https://github.com/hughperkins/DeepCL',
    long_description=read('README.rst'),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
    ],
    install_requires=[],
    scripts=['test_deepcl.py', 'test_lowlevel.py'],
    ext_modules=ext_modules,
)
