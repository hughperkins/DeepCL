  DeepCL
==========

Global Contents
===============

- [This page](doc/README.md)
- [Command line usage](doc/Commandline.md)
- [Neural Net API](doc/NeuralNetAPI.md)
- [Python wrapper](python/README.md)
- [Q-learning](doc/QLearning.md)
- [To build](doc/Build.md)
- [Testing](doc/Testing.md)
- [Architecture](doc/Architecture.md)
- [Changes](doc/Changes.md)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [DeepCL](#deepcl)
- [To use the pre-built binaries](#to-use-the-pre-built-binaries)
  - [What if it doesn't run?](#what-if-it-doesnt-run)
- [What if I need a new feature?](#what-if-i-need-a-new-feature)
- [What if I want to contribute myself?](#what-if-i-want-to-contribute-myself)
  - [Development technical details](#development-technical-details)
  - [Architecture](#architecture)
- [Third-party libraries](#third-party-libraries)
- [Related projects](#related-projects)
- [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

DeepCL
==========

OpenCL library to train deep convolutional networks
- C++
- OpenCL
- Deep convolutional
- (New!) includes Q-learning module (draft)
- (New!) Python wrappers available (draft too :-) )

Functionalities:
* convolutional layers
* max-pooling
* normalization layer
* random translations, as in [Flexible, High Performance Convolutional Neural Networks for Image Classification](http://ijcai.org/papers11/Papers/IJCAI11-210.pdf)
* random patches, as in [ImageNet Classification with Deep Convolutional Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* multinet, ie Multi-column deep convolutional network, [McDnn](http://arxiv.org/pdf/1202.2745.pdf)
* simple command-line network specification, as per notation in [Multi-column Deep Neural Networks for Image Classification](http://arxiv.org/pdf/1202.2745.pdf)
* pad-zeros possible for convolutional layer
* various activation functions available:
  * tanh
  * scaled tanh (1.7519 * tanh(2/3x) )
  * linear
  * sigmoid
  * relu
  * softmax
* fully-connected layers
* various loss layers available:
  * square loss
  * cross-entropy
  * multinomial cross-entropy (synonymous with multinomial logistic, etc)
* Q-learning

Example usage:
- intend to target 19 x 19 Go boards, eg something similar to [Clark and Storkey](http://arxiv.org/abs/1412.3409) or [Maddison, Huang, Sutskever and Silver](http://arxiv.org/abs/1412.6564)
  - obtained 36.3% test accuracy, on next move prediction task, using 33.6 million training examples from [kgsgo v2 dataset](https://github.com/hughperkins/kgsgo-dataset-preprocessor)
  - commandline used `./deepclrun dataset=kgsgoall netdef=32c5{z}-32c5{z}-32c5{z}-32c5{z}-32c5{z}-32c5{z}-500n-361n numepochs=3 learningrate=0.0001`
  - 3 epochs, 1.5 days per epoch, on an Amazon GPU instance, comprising half an NVidia GRID K520 GPU (about half as powerful as a GTX780)
- obtained 99.5% test accuracy on MNIST, using `netdef=rt2-8c5{padzeros}-mp2-16c5{padzeros}-mp3-150n-10n numepochs=20 multinet=6 learningrate=0.002`
  - epoch time 99.8 seconds, using an Amazon GPU instance, ie half an NVidia GRID K520 GPU (since we are learning 6 nets in parallel, so 16.6seconds per epoch per net)

For Python wrappers, please see [python/README.md](python/README.md)

# To use the pre-built binaries

## Python

* For python, please use [pip](https://pypi.python.org/pypi/DeepCL/3.4.1)

## Windows

Pre-built binaries are available for Windows.  In order to use them you need:
* An OpenCL driver for your GPU
* A recent release with Windows binaries is [v3.4.1](https://github.com/hughperkins/DeepCL/releases/tag/v3.4.1)

## Linux

Pre-built binaries are available for linux.  In order to use them you need:
* An OpenCL driver for your GPU
* A recent release with linux binaries is [v3.4.1](https://github.com/hughperkins/DeepCL/releases/tag/v3.4.1)

## What if it doesn't run?

* Check if you have an OpenCL-enabled device on your system
  * ideally a GPU, or accelerator, since there is no attempt to optimize DeepCL for CPUs (at least, not currently, could change, feel free to submit a pull request :-) )
* Try running `gpuinfo` (from [OpenCLHelper](https://github.com/hughperkins/OpenCLHelper), but built as part of this project too, for ease of use )
  * it should output at least one OpenCL-enabled device
  * if it doesn't, then you need to make sure you have an OpenCL-enabled device, and that appropriate drivers are installed, and that the ICD is configured appropriately (registry in Windows, and `/etc/OpenCL/vendors` in linux)

# What if I need a new feature?

Please raise an issue, let me know you're interested.
* If it's on my list of things I was going to do sooner or later anyway (see below), I might do it sooner rather than later.
* If it's to do with usability, I will try to make that a priority

What if I want to contribute myself?
=================

- please feel free to fork this repository, tweak things, send a pull request

## Development technical details
* [cogapp](http://nedbatchelder.com/code/cog/) generator is used extensively, to accelerate development, reduce the number of manual copy-and-pasting and so on.  Specifically, it's used for:
  * generating header declarations from .cpp definition files
  * generating fluent-style argument classes for certain tests
  * ... and more uses will surely be found :-)
* You need Python installed and available for this to work.  You don't need python just to
build the sources, but if you do have python installed, and you flip the `PYTHON_AVAILABLE` switch in the 
cmake configuration, then a lot of manual editing will no longer be necessary :-)

Third-party libraries
=====================

* [OpenCLHelper](https://github.com/hughperkins/OpenCLHelper)
* [clew](https://github.com/martijnberger/clew)
* [libpng++](http://www.nongnu.org/pngpp/doc/0.2.1/)

Related projects
================

* [kgsgo-dataset-preprocessor](https://github.com/hughperkins/kgsgo-dataset-preprocessor) Dataset based on kgsgo games; 33 million data points

Credits
=======

Thank-you very much to Tambet Matilsen for his assistance in morphing the project into a somewhat presentable state, and for everyone who has clicked on the 'star' button above :-)

License
=======

[Mozilla Public License 2.0](http://mozilla.org/MPL/2.0/)


