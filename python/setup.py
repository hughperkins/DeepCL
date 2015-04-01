#from distutils.core import setup
import os
import os.path
import sysconfig
import sys
import glob
import platform
from setuptools import setup
#from distutils.extension import Extension
from setuptools import Extension

cython_present = False
try:
    from Cython.Build import cythonize
    cython_present = True
except ImportError:
    pass

pypandoc_present = False
try:
    import pypandoc
    pypandoc_present = True
except ImportError:
    pass

print ( sys.argv )

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def get_so_suffix():
    if sysconfig.get_config_var('SOABI') != None:
        return "." + sysconfig.get_config_var('SOABI')
    return ""

#def read_md( mdname ): 
#    if pypandoc_present:
#        return pypandoc.convert(mdname, 'rst')
#    else:
#        print("warning: pypandoc module not found, could not convert Markdown to RST")
#        return open(mdname, 'r').read()

if pypandoc_present:
    pypandoc.convert('README.md', 'rst', outputfile = 'README.rst' )

def my_cythonize(extensions, **_ignore):
    #newextensions = []
    for extension in extensions:
        print(extension.sources)
        should_cythonize = False
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                should_cythonize = True
                if not cython_present:
                    if extension.language == 'c++':
                        ext = '.cpp'
                    else:
                        ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        print(should_cythonize)
        if should_cythonize and cython_present:
            print('cythonizing...')
            cythonize(extension)
        extension.sources[:] = sources    
        #newextensions.append( extension )
    return extensions

def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources    
    return extensions

# from http://stackoverflow.com/questions/14320220/testing-python-c-libraries-get-build-path
def distutils_dir_name(dname):
    """Returns the name of a distutils build directory"""
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)
 
def lib_build_dir():
    return os.path.join('build', distutils_dir_name('lib'))

clconvolve_sourcestring = """LayerMaker.cpp NeuralNetMould.cpp
     ConvolutionalLayer.cpp NeuralNet.cpp Layer.cpp InputLayer.cpp
    Propagate1.cpp Propagate.cpp Propagate2.cpp Propagate3.cpp LayerDimensions.cpp
    Propagate4.cpp ActivationFunction.cpp SquareLossLayer.cpp LossLayer.cpp BackpropWeights2.cpp
    BackpropWeights2Cpu.cpp BackpropErrorsv2.cpp BackpropErrorsv2Cpu.cpp
    BackpropWeights2Naive.cpp BackpropErrorsv2Naive.cpp BackpropWeights2Scratch.cpp
    CrossEntropyLoss.cpp SoftMaxLayer.cpp FullyConnectedLayer.cpp  EpochMaker.cpp
    PoolingPropagate.cpp PoolingPropagateCpu.cpp PoolingLayer.cpp PoolingBackprop.cpp
    PoolingBackpropCpu.cpp PoolingPropagateGpuNaive.cpp BackpropWeights2ScratchLarge.cpp
    BatchLearner.cpp NetdefToNet.cpp NetLearner.cpp stringhelper.cpp NormalizationLayer.cpp
    RandomPatches.cpp RandomTranslations.cpp NorbLoader.cpp MultiNet.cpp
    Trainable.cpp InputLayerMaker.cpp ConvolutionalMaker.cpp RandomTranslationsMaker.cpp
    RandomPatchesMaker.cpp NormalizationLayerMaker.cpp FullyConnectedMaker.cpp
    PoolingMaker.cpp PatchExtractor.cpp Translator.cpp GenericLoader.cpp Kgsv2Loader.cpp
    BatchLearnerOnDemand.cpp NetLearnerOnDemand.cpp BatchProcess.cpp WeightsPersister.cpp
    PropagateFc.cpp BackpropErrorsv2Cached.cpp PropagateByInputPlane.cpp
    PropagateExperimental.cpp PropagateAuto.cpp PropagateCpu.cpp Propagate3_unfactorized.cpp
    PoolingBackpropGpuNaive.cpp""" 
clconvolve_sources_all = clconvolve_sourcestring.split()
clconvolve_sources = []
for source in clconvolve_sources_all:
    clconvolve_sources.append(source)

openclhelpersources = list(map( lambda name : '../' + name, [ 'OpenCLHelper/OpenCLHelper.cpp',
        'OpenCLHelper/deviceinfo_helper.cpp', 'OpenCLHelper/platforminfo_helper.cpp',
        'OpenCLHelper/CLKernel.cpp', 'OpenCLHelper/thirdparty/clew/src/clew.c' ] ))
print(openclhelpersources)
print(isinstance( openclhelpersources, list) )

compile_options = []
osfamily = platform.uname()[0]
if osfamily == 'Windows':
   compile_options.append('/EHsc')
elif osfamily == 'Linux':
   compile_options.append('-std=c++11')
else:
   pass
   # put other options etc here if necessary

#if osfamily == 'Windows' and sys.version_info[0] == 2:
#    print('WARNING: python 2.x not really supported, since it needs visual studio 9; and visual studio 2009 doesnt support any c++11 features, which we would ideally prefer to be able to use')
#    print('Probably possible to coerce ClConvolve to work with visual studio 9, and by extension with python 2.x, but maybe easier to just use python 3.x instead?')

runtime_library_dirs = []
libraries = []
if osfamily == 'Linux':
    runtime_library_dirs= ['.']

if osfamily == 'Windows':
    libraries = ['winmm']

if cython_present:
    my_cythonize = cythonize
else:
    my_cythonize = no_cythonize

#libraries = [
#    ("OpenCLHelper", {
#        'sources': openclhelpersources + ['dummy_openclhelper.cpp'],
#        'include_dirs': ['ClConvolve/OpenCLHelper'],
#        'extra_compile_args': compile_options,
##        define_macros = [('OpenCLHelper_EXPORTS',1)],
##        libraries = []
##        language='c++'
#        }
#    )
#]

ext_modules = [
#    Extension("_OpenCLHelper",
#        sources = openclhelpersources + ['dummy_openclhelper.cpp'],
#        include_dirs = ['ClConvolve/OpenCLHelper'],
#        extra_compile_args=compile_options,
#        define_macros = [('OpenCLHelper_EXPORTS',1),('MS_WIN32',1)],
##        libraries = []
##        language='c++'
#    )
#    Extension("libClConvolve",
#        list(map( lambda name : 'ClConvolve/src/' + name, clconvolve_sources)), # +
##            glob.glob('ClConvolve/src/*.h'),
#        include_dirs = ['ClConvolve/src','ClConvolve/OpenCLHelper'],
#        extra_compile_args = compile_options,
#        library_dirs = [ lib_build_dir() ],
#        libraries = [ "OpenCLHelper" + get_so_suffix() ],
#        define_macros = [('ClConvolve_EXPORTS',1)],
#        runtime_library_dirs=runtime_library_dirs
##        language='c++'
#    ),
    Extension("PyClConvolve",
              sources=["PyClConvolve.pyx", 'CyWrappers.cpp'] 
                + openclhelpersources
                + list(map( lambda name : '../src/' + name, clconvolve_sources))
                + ['../qlearning/QLearner.cpp','../qlearning/array_helper.cpp'], 
#                glob.glob('ClConvolve/OpenCLHelper/*.h'),
              include_dirs = ['../src','../OpenCLHelper','../qlearning'],
              libraries= libraries,
              extra_compile_args=compile_options,
        define_macros = [('ClConvolve_EXPORTS',1),('OpenCLHelper_EXPORTS',1)],
#              extra_objects=['cClConvolve.pxd'],
#              library_dirs = [lib_build_dir()],
              runtime_library_dirs=runtime_library_dirs,
              language="c++"
    )
]

setup(
  name = 'DeepCL',
  version = "0.0.7",
  author = "Hugh Perkins",
  author_email = "hughperkins@gmail.com",
  description = 'python wrapper for ClConvolve deep convolutional neural network library for OpenCL',
  license = 'MPL',
  url = 'https://github.com/hughperkins/PyClConvolve',
  long_description = read('README.rst'),
  classifiers = [
    'Development Status :: 4 - Beta',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
  ],
  install_requires = ['Cython>=0.22','cogapp>=2.4','future>=0.14.3'],
  tests_require = ['nose>=1.3.4'],
  scripts = ['test_clconvolve.py','test_lowlevel.py'],
 # modules = libraries,
#  libraries = libraries,
  ext_modules = my_cythonize( ext_modules),
)

