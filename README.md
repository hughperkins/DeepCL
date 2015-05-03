  DeepCL
==========

- [Python API](python/README.md)
- [Lua API](lua/README.md)
- [Command line API](doc/Commandline.md)
- [C++ API](doc/NeuralNetAPI.md)
- [Q-learning](doc/QLearning.md)
- [To build](doc/Build.md)
- [Stable API](doc/PublicApis.md)
- [Development](doc/Development.md)
- [Benchmarking](doc/Benchmarking.md)
- [Changes](doc/Changes.md)
- [Doc for old 3.x.x](https://github.com/hughperkins/DeepCL/blob/3.x.x/README.md)

DeepCL
==========

OpenCL library to train deep convolutional networks
- C++
- OpenCL
- Deep convolutional
- Python wrappers
- Lua wrappers
- Q-learning

APIs:
* [Python](python/README.md)
* [Lua](lua/README.md)
* [c++](doc/NeuralNetAPI.md)
* [command-line](doc/Commandline.md)

Layer types:
* convolutional
* max-pooling
* normalization
* activation
* dropout
* random translations
* random patches
* loss

Loss layer types:
* softmax
* cross-entropy (synonymous with multinomial logistic, etc)
* square loss

Trainers:
* SGD (including momentum)
* Anneal (New!)
* Nesterov (New!)
* Adagrad (New!)
* Rmsprop (New!)

Activations:
* tanh
* scaled tanh (1.7519 * tanh(2/3x) )
* linear
* sigmoid
* relu

Multicolumn net also possible, as in [McDnn](http://arxiv.org/pdf/1202.2745.pdf)

# To install

## Python

* For python, please use [Python API](python/README.md), or use [pip](https://pypi.python.org/pypi/DeepCL)

## Lua

* For Lua, please see [Lua API](lua/README.md), or use [luarocks](https://luarocks.org/modules/hughperkins/luadeepcl)

## Commandline tools, and c++ libraries

### Windows

Pre-built binaries are available for Windows.  In order to use them you need:
* An OpenCL driver for your GPU
* A recent release with Windows binaries is [v5.5.0](https://github.com/hughperkins/DeepCL/releases/tag/v5.5.0) 

### linux

For linux, please [build from source](doc/Build.md)

## What if it doesn't run?

* Check if you have an OpenCL-enabled device on your system
  * ideally a GPU, or accelerator, since there is no attempt to optimize DeepCL for CPUs (at least, not currently, could change, feel free to submit a pull request :-) )
* Try running `gpuinfo` (from [EasyCL](https://github.com/hughperkins/EasyCL), but built as part of this project too, for ease of use )
  * it should output at least one OpenCL-enabled device
  * if it doesn't, then you need to make sure you have an OpenCL-enabled device, and that appropriate drivers are installed, and that the ICD is configured appropriately (registry in Windows, and `/etc/OpenCL/vendors` in linux)

# What if I need a new feature?

Please raise an issue, let me know you're interested.
* If it's on my list of things I was going to do sooner or later anyway (see below), I might do it sooner rather than later.
* If it's to do with usability, I will try to make that a priority

What if I want to contribute myself?
=================

- please feel free to fork this repository, tweak things, send a pull request.  Or get in contact. Or both :-)

Third-party libraries
=====================

* [EasyCL](https://github.com/hughperkins/EasyCL)
* [clew](https://github.com/martijnberger/clew)
* [libpng++](http://www.nongnu.org/pngpp/doc/0.2.1/)

Related projects
================

* [kgsgo-dataset-preprocessor](https://github.com/hughperkins/kgsgo-dataset-preprocessor) Dataset based on kgsgo games; 33 million data points

Credits
=======

* Tambet Matilsen has provided excellent suggestions and feedback on which functionalities to prioritize, and on how to make the website somewhat presentable

License
=======

[Mozilla Public License 2.0](http://mozilla.org/MPL/2.0/)

To get in contact
=================

There is a mailing list at http://lists.hughperkins.com/listinfo.cgi/deepcl-hughperkins.com for discussions, ideas, or just to say 'hi'.  You can also just create issues, in github, in the top right of this page.

