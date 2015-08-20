# Python Wrapper Development

## Notes on how the wrapper works

* [cDeepCL.pxd](https://github.com/hughperkins/DeepCL/blob/master/python/cDeepCL.pxd) contains the definitions of the underlying DeepCL c++ libraries classes
* [PyDeepCL.pyx](https://github.com/hughperkins/DeepCL/blob/master/python/PyDeepCL.pyx) contains Cython wrapper classes around the underlying c++ classes
* [setup.py](https://github.com/hughperkins/DeepCL/blob/master/python/setup.py) is a setup file for compiling the `PyDeepCL.pyx` Cython file

## Maintainer/development information

If you want to modify the python wrappers, you'll need to re-run Cython.  This is no longer handled by `setup.py`, but is handled by the cmake build.  So, to run cython you'll need to:
- install Cython, eg `pip install cython`
- follow the instructions for the native build, [Build.md](https://github.com/hughperkins/DeepCL/blob/8.x/doc/Build.md)
- when you open `ccmake`:
  - enable option `Maintainer options`, then press `c`/`configure`
  - enable `BUILD_PYTHON_WRAPPERS`, then `c`/`configure`
  - enable `DEV_RUN_CYTHON`, then `c`/`configure`
- => and now `g`/`generate`, and build

* If you want to update this readme, you need to re-generate the README.rst, so you'll need pypandoc:
```
pip install pypandoc
```
  * (note that pypandoc depends on pandoc native library)

And then to regenerate README.rst:
```
python setup.py sdist
```

