  DeepCL
==========

- [Python API](python/README.md)
- [Lua API](lua/README.md)
- [Command line API](doc/Commandline.md)
- [C++ API](doc/NeuralNetAPI.md)
- [Q-learning](doc/QLearning.md)
- [To build](doc/Build.md)
- [Development](doc/Development.md)
- [Changes](doc/Changes.md)

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
* Adadelta (New!)

Activations:
* tanh
* scaled tanh (1.7519 * tanh(2/3x) )
* linear
* sigmoid
* relu

[Loader formats](doc/Loaders.md):
* jpegs
* mnist
* kgsv2
* norb

Weight initializers:
* original
* uniform
* more possible...

Multicolumn net also possible, as in [McDnn](http://arxiv.org/pdf/1202.2745.pdf)

# Example usages

- obtained 37.2% test accuracy, on next move prediction task, using 33.6 million training examples from [kgsgo v2 dataset](https://github.com/hughperkins/kgsgo-dataset-preprocessor)
  - commandline used `./deepclrun dataset=kgsgoall netdef=12*(32c5z-relu)-500n-tanh-361n numepochs=15 learningrate=0.0001`
  - 2 epochs, 2 days per epoch, on an Amazon GPU instance, comprising half an NVidia GRID K520 GPU (about half as powerful as a GTX780)
- obtained 99.5% test accuracy on MNIST, using `netdef=rt2-8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-10n numepochs=20 multinet=6 learningrate=0.002`
   - epoch time 99.8 seconds, using an Amazon GPU instance, ie half an NVidia GRID K520 GPU (since we are learning 6 nets in parallel, so 16.6seconds per epoch per net)
 
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

Pre-build binaries are available for linux.  In order to use them you need:
* An OpenCL driver for your GPU
* A recent release with linux binaries is [v5.5.0](https://github.com/hughperkins/DeepCL/releases/tag/v5.5.0) 

If the binaries dont work on your distribution, please [build from source](doc/Build.md)

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
* lua
* cogapp

Related projects
================

* [kgsgo-dataset-preprocessor](https://github.com/hughperkins/kgsgo-dataset-preprocessor) Dataset based on kgsgo games; 33 million data points

Credits
=======

* Tambet Matilsen has provided excellent suggestions and feedback on which functionalities to prioritize, and on how to make the website somewhat presentable

License
=======

[Mozilla Public License 2.0](http://mozilla.org/MPL/2.0/)

Recent changes
==============

* June 22nd:
  * removed lua wrappers
  * if you want to use lua with OpenCL, please consider using [cltorch](http://github.com/hughperkins/cltorch) and [clnn](http://github.com/hughperkins/clnn)

To get in contact
=================

There is a mailing list at http://lists.hughperkins.com/listinfo.cgi/deepcl-hughperkins.com for discussions, ideas, or just to say 'hi'.  You can also just create issues, in github, in the top right of this page.

