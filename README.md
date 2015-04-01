  ClConvolve
==========

Global Contents
===============

- [This page](README.md)
- [Command line usage](Commandline.md)
- [Neural Net API](NeuralNetAPI.md)
- [Python wrapper](PyClConvolve/README.md)
- [Formulae notes](Formulae.md)
- [To build](Build.md)
- [Testing](Testing.md)
- [Changes](Changes.md)

Page Contents
========

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [ClConvolve](#clconvolve)
- [Validation against standard datasets](#validation-against-standard-datasets)
  - [NORB](#norb)
  - [MNIST](#mnist)
- [Q-learning (draft)](#q-learning-draft)
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

ClConvolve
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
  - commandline used `./clconvolve1 dataset=kgsgoall netdef=32c5{z}-32c5{z}-32c5{z}-32c5{z}-32c5{z}-32c5{z}-500n-361n numepochs=3 learningrate=0.0001`
  - 3 epochs, 1.5 days per epoch, on an Amazon GPU instance, comprising half an NVidia GRID K520 GPU (about half as powerful as a GTX780)
- obtained 99.5% test accuracy on MNIST, using `netdef=rt2-8c5{padzeros}-mp2-16c5{padzeros}-mp3-150n-10n numepochs=20 multinet=6 learningrate=0.002`
  - epoch time 99.8 seconds, using an Amazon GPU instance, ie half an NVidia GRID K520 GPU (since we are learning 6 nets in parallel, so 16.6seconds per epoch per net)

For Python wrappers, please see [PyClConvolve/README.md](PyClconvolve/README.md)


# Validation against standard datasets

## NORB

* Download the data files from [NORB datafiles](http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/), and place in `data/norb`, decompressed, ie (g)unzipped
* Pre-process, to shuffle the training samples, and draw 1000 testing samples:
```bash
./prepare-norb ../data/norb
```
* Run training, eg, based on LeCun's lenet-7:
```bash
./clconvolve1 netdef=8C5-MP4-24C6-MP3-80C6-5N learningrate=0.00001 dataset=norb
```
* On an Amazon AWS GPU instance, which has an NVidia GRID K520 GPU, this has epoch time of 76 seconds, and reaches test accuracy of around 91.7% after around 200 epochs (train accuracy 99.996%!)

## MNIST

* You can download the MNIST data from [MNIST database](http://yann.lecun.com/exdb/mnist/) , and place in the `data\mnist` directory, (g)unzipped.
* Convert from idx to mat format:
```bash
./idx-to-mat ../data/mnist train
./idx-to-mat ../data/mnist t10k
```
* Run as per the [convnetjs MNIST demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html) architecture as follows:
```bash
./clconvolve1 netdef=8c5{padzeros}-mp2-16c5{padzeros}-mp3-10n learningrate=0.002 dataset=mnist
```
* On an Amazon AWS GPU instance, epoch time is about 13.8 seconds, giving about 98.7% test accuracy, after 12 epochs
* Actually, I think the following gives slightly better test accuracy, about 99.0%, using 17.2seconds per epoch:
```bash
./clconvolve1 netdef=8c5{padzeros}-mp2-16c5{padzeros}-mp3-150n-10n learningrate=0.002 dataset=mnist
```

# Q-learning (draft)

* Started to write a q-learning module
* Ideally, this will be callable from Python in the future, via [PyClConvolve](https://pypi.python.org/pypi/PyClConvolve)
* Look at [ScenarioImage.h](prototyping/qlearning/ScenarioImage.h) and [ScenarioImage.cpp](prototyping/qlearning/ScenarioImage.cpp) for an example scenario we can feed to it
  * [learnScenarioImage.cpp](prototyping/qlearning/learnScenarioImage.cpp) is a corresponding example of how we can learn this
* The qlearning module is at [QLearner.h](qlearning/QLearner.h), and the interface for scenarios at [Scenario.h](qlearning/Scenario.h)

# To use the pre-built binaries

Pre-built binaries are available for Windows-64, for certain releases.  In order to use them you need:
* [Windows 2013 redistributable](http://www.microsoft.com/en-us/download/details.aspx?id=40784).
* An OpenCL driver for your GPU
* A recent release with Windows binaries is [v2.0.2](https://github.com/hughperkins/ClConvolve/releases/tag/v2.0.2)

## What if it doesn't run?

* Check if you have an OpenCL-enabled device on your system
  * ideally a GPU, or accelerator, since there is no attempt to optimize ClConvolve for CPUs (at least, not currently, could change, feel free to submit a pull request :-) )
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

## Architecture

* [NeuralNet.h](src/NeuralNet.h) is a container for layers. It contains three types of method:
  * methods that iterate over each layer, eg `propagate`
  * methods that call a method on the first layer, eg `getInputCubeSize`
  * methods that call a method on the last layer, eg `getResults()`
* Various net layers, eg [ConvolutionalLayer.cpp](src/ConvolutionalLayer.cpp), [PoolingLayer.cpp](src/PoolingLayer.cpp), etc
* Trying to debug/unit-test by training whole layers is challenging, so the layer implementations are factorized, over two levels.  The first level abstracts away propagation, backprop of errors, and backprop of weights:
  * [Propagate.cpp](src/Propagate.cpp) handles forward propagation
  * [BackpropErrorsv2.cpp](src/BackpropErrorsv2.cpp) handles backward propagation of errors (strictly speaking: of the partial derivative of the loss with respect to the pre-activation sums for the layer)
    * The results of this layer are passed back through the stack of layers
  * [BackpropWeights2.cpp](src/BackpropWeights2.cpp) handles backward propagation of weights, from the results of the appropriate BackpropErrorsv2 layer
* Then, each of these classes calls into implementation classes, which are children of the same class, which provide various kernels and implementations.  Eg, for [Propagate.h](src/Propagate.h], we have:
  * [Propagate1.cpp](src/Propagate1.cpp)
  * [Propagate2.cpp](src/Propagate2.cpp)
  * [Propagate3.cpp](src/Propagate3.cpp)
  * ...
* ... and similarly for [BackpropErrorsv2](src/BackpropErrorsv2.cpp), and [BackpropWeights2.cpp](src/BackpropWeights2.cpp): each has implementation classes
* Therefore:
  * Testing can target one single implementation, or target only propagate or backproperrors, or backpropweights, rather than needing to test an entire network
  * These lower level factorized implementations could also plausibly be an appropriate unit of re-use
* There are also "meta"-layers, ie:
  * [PropagateAuto.cpp](src/PropagateAuto.cpp): automatically tries different propagate kernels at run-time, and chooses the fastest :-)

Third-party libraries
=====================

* [OpenCLHelper](https://github.com/hughperkins/OpenCLHelper)
* [clew](https://github.com/martijnberger/clew)
* [libpng++](http://www.nongnu.org/pngpp/doc/0.2.1/)

Related projects
================

* [kgsgo-dataset-preprocessor](https://github.com/hughperkins/kgsgo-dataset-preprocessor) Dataset based on kgsgo games; 33 million data points

License
=======

[Mozilla Public License 2.0](http://mozilla.org/MPL/2.0/)


