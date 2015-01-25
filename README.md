ClConvolve
==========

Contents
========

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [ClConvolve](#clconvolve)
- [Validation against standard datasets](#validation-against-standard-datasets)
  - [NORB](#norb)
  - [MNIST](#mnist)
    - [Data](#data)
    - [Architecture from convjs](#architecture-from-convjs)
    - [Architecture from lenet5](#architecture-from-lenet5)
    - [Uniform layers](#uniform-layers)
    - [Detailed results](#detailed-results)
- [Neural Net API](#neural-net-api)
  - [Create a net](#create-a-net)
  - [Add some layers](#add-some-layers)
  - [Train](#train)
  - [Test](#test)
- [Data format](#data-format)
- [Pre-requisites](#pre-requisites)
- [To build](#to-build)
- [Linking](#linking)
- [Correctness checking](#correctness-checking)
- [Unit-testing](#unit-testing)
  - [Concepts](#concepts)
  - [Implementation](#implementation)
- [Formulae notes](#formulae-notes)
- [Development notes](#development-notes)
- [What's done / what's planned](#whats-done--whats-planned)
- [Recent changes](#recent-changes)
- [Third-party libraries](#third-party-libraries)
- [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

ClConvolve
==========

OpenCL library to train deep convolutional networks
- C++
- OpenCL
- Deep convolutional

Target usage:
- 19 x 19 Go boards, eg something similar to [Clark and Storkey](http://arxiv.org/abs/1412.3409) or [Maddison, Huang, Sutskever and Silver](http://arxiv.org/abs/1412.6564)
- Also works on MNIST 28 x 28 boards
  - obtained 98.6% test accuracy on MNIST, ~~using 2 convolutional layers of 32 filters, each filter 5 by 5, and with zero-padding applied~~, using the architecture given on [convjs mnist demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)
- Tested on [NORB](http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/) dataset, gets 90% at the moment, which is not too far away from [LeCun's results](http://yann.lecun.com/exdb/publis/pdf/lecun-04.pdf).

# Validation against standard datasets

## NORB

* Create a network like this:
```c++
    NeuralNet *net = NeuralNet::maker()->planes(numPlanes)->boardSize(boardSize)->instance();
    net->convolutionalMaker()->numFilters(8)->filterSize(5)->relu()->biased()->insert();
    net->poolingMaker()->poolingSize(4)->insert();
    net->convolutionalMaker()->numFilters(24)->filterSize(6)->relu()->biased()->insert();
    net->poolingMaker()->poolingSize(3)->insert();
    net->fullyConnectedMaker()->numPlanes(5)->boardSize(1)->linear()->biased()->insert();
    net->softMaxLossMaker()->insert();
```
* I think this is missing a layer actually, might need some tweaking possibly
* Current results is about 90% test accuracy, after 20 epochs.  Each epoch is 125seconds
* To run
  * First download the [norb datafiles](http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/) to `data/norb`, and gunzip them
  * Run pre-processing:
```bash
make
./prepare-norb
```
* pre-processing done is:
  * shuffle training samples
  * draw 1000 samples from test set, which is enough to determine test accuracy to nearest 0.1%
* Then run:
```bash
./testnorb1
```

## MNIST

### Data

* Please download from [MNIST database](http://yann.lecun.com/exdb/mnist/) , and place in the `data\mnist` directory, (g)unzipped.

### Architecture from convjs

* Based on [convjs MNIST demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html) architecture
* Create a network like this:
```c++
NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(28)->instance();
net->convolutionalMaker()->numFilters(8)->filterSize(5)->relu()->biased()->padZeros()->insert();
net->poolingMaker()->poolingSize(2)->insert();
net->convolutionalMaker()->numFilters(16)->filterSize(5)->relu()->biased()->padZeros()->insert();
net->poolingMaker()->poolingSize(3)->insert();
net->fullyConnectedMaker()->numPlanes(10)->boardSize(1)->linear()->biased()->insert();
net->softMaxLossMaker()->insert();
net->setBatchSize(128);
```
* Compared to the convjs demo:
  * convjs demo augments the data, by cropping a 24x24 region, which we're not doing here
* For the implementation above, on my tests, using an Amazon AWS GPU instance, which has an NVidia GRID K520 GPU, epoch time was 13.8seconds, and test accuracy was around 98.7%, using ClConvolve v0.5, after 12 epochs
* Implementation: [test/testmnist-convjs.cpp](test/testmnist-convjs.cpp)

### Architecture from lenet5

* Based on [LeCun's paper](http://yann.lecun.com/exdb/publis/index.html#lecun-98)
* Create a net as follows:
```c++
NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(32)->instance();
net->convolutionalMaker()->numFilters(6)->filterSize(5)->relu()->biased()->insert();
net->poolingMaker()->poolingSize(2)->insert();
net->convolutionalMaker()->numFilters(16)->filterSize(5)->relu()->biased()->insert();
net->poolingMaker()->poolingSize(2)->insert();
net->fullyConnectedMaker()->numPlanes(10)->boardSize(1)->linear()->biased()->insert();
net->softMaxLossMaker()->insert();
```
* this is similar to lenet-5, but not quite the same, specifically:
  * lenet-5 has RBF layers at the end (page 8, second column, of the paper)
  * lenet-5 has multiple of these RBF and fully-connected layers at the end
  * lenet-5 uses tanh (I think?), not relu
  * lenet-5 is not using max-pooling but something more like average-pooling, and it has an activation function applied (sigmoid) (page 7, second column, bottom half)
  * lenet-5 is not connecting all filters from one layer with all filters of the next layer (Table I, of the paper)
* Note that the images are padded with a border of margin 2 by the implementation, as per lenet5
* Using an Amazon AWS GPU instance, which has an NVidia GRID K520 GPU, epoch time was 14.6seconds, and test accuracy was around 98.7%, using ClConvolve v0.5, after 12 epochs
* Implementation: [test/testmnist-lenet5.cpp](test/testmnist-lenet5.cpp)

### Uniform layers

* [test/testmnist.cpp](test/testmnist.cpp) creates multiple configurable uniform layers as follows:
```c++
NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(28)->instance();
for( int layer = 0; layer < numlayers; layer++ ) {
    net->convolutionalMaker()->numFilters(numfilters)->filterSize(filtersize)->relu()->biased(biased)->padZeros(padzeros)->insert();
}
net->fullyConnectedMaker()->numPlanes(10)->boardSize(1)->linear()->biased()->insert();
net->softMaxLossMaker()->insert();
net->setBatchSize(128);
```
* This doesn't run as quickly, or give as good results, as the highly-optimized networks listed earlier, but might be interesting for easy experimentation, with different layer depths, filter sizes, and so on?

### Detailed results

* Following results on MNIST, using an Amazon AWS GPU instance, which has an NVidia GRID K520 GPU:

| Test accuracy| Number epochs | Epoch time (s) | Number filter layers| Filters per layer | Filter size | Pad zeros | Version| Learning rate  |
|-------|----|---------------|--------------|----------|------------|-------------|---|---------------|
| 97.5%| 12| 17.2 | 1| 32 | 5  | No  | v0.3(1) | 0.0001  |
| 98.1%| 20| 17.2 | 1| 32 | 5 | No| v0.3(1) | 0.0001     |
| 98.3%| 12| 80.5 | 2 | 32 | 5 | No | v0.3(1) | 0.0001   |
| 98.5%| 15| 80.5 | 2| 32 | 5 | No  | v0.3(1) | 0.0001   |
| 98.57% +/- 0.03%| 20| | 3  | 32 | 5 | No | v0.3(1) | 0.0001  |
| 98.64% +/- 0.02%| 20| 203 | 2 | 32 | 5 | Yes| v0.3(1) | 0.0001    |
| 98.7% +/- 0.1% | 12 | 13.8s | 2 conv 2 pool | 8,16 | conv:5,5 pool:2,3 | Yes | next v0.7 (2) | 0.002 |
| 98.7% +/- 0.1% | 12 | 14.6s | 2 conv 2 pool | 6,16 | conv:5,5 pool:2,2 | Yes | next v0.7 (3) | 0.002 |

* Notes:
  * (1) Using [testmnist](test/testmnist.cpp) or `testmnist-softmax`
  * (2) Using [testmnist-convjs](test/testmnist-convjs.cpp)
  * (3) Using [testmnist-lenet5](test/testmnist-lenet5.cpp)

Neural Net API
==============

Create a net
-------------

```c++
#include "ClConvolve.h"

NeuralNet *net = NeuralNet::maker()->planes(10)->boardSize(19)->instance();
```

* You can change the number of input planes, and the board size.

Add some layers
---------------

*Convolutional layers*

Eg:
```c++
net->ConvolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
```

* You can change the number of filters, and their size.  If you want, you can use any of the following options:
  * `->padZeros()`: pad the input board with zeros, so the output board is same size as the input
  * `->biased()` turn on bias
  * `->biased(1)` same as `->biased()`
  * `->biased(0)` turn off bias (default)
  * `->linear()` choose linear activation
  * `->relu()` choose relu activation
  * `->sigmoid()` choose sigmoid activation
  * `->tanh()` choose tanh activation (current default, but defaults can change...)
* convolutional layers forward-prop and backward-prop both run on GPU, via OpenCL

*Fully connected layers*

eg:
```c++
net->fullyConnectedMaker()->numPlanes(2)->boardSize(28)->insert();
```

Available options:
  * `->biased()` turn on bias
  * `->biased(1)` same as `->biased()`
  * `->biased(0)` turn off bias (default)
  * `->linear()` choose linear activation
  * `->relu()` choose relu activation
  * `->sigmoid()` choose sigmoid activation
  * `->tanh()` choose tanh activation (current default, but defaults can change...)

*Max-pooling layers*

```c++
net->poolingMaker()->poolingSize(2)->insert();
```
* By default, if the input board size is not an exact multiple of the poolingsize, the extra margin will be ignored
* You can specify `padZeros` to include this margin:
```c++
net->poolingMaker()->poolingSize(2)->padZeros()->insert();
```

*Loss layer*

You need to add exactly one loss layer, as the last layer of the net.  The following loss layers are available:
```c++
net->squareLossMaker()->insert();
net->crossEntropyLossMaker()->insert();
net->softMaxLossLayer()->insert();
```
* if your outputs are categorial, 1-of-N, then softMaxLossLayer is probably what you want
* otherwise, you can choose square loss, or cross-entropy loss:
  * squared loss works well with a `tanh` last layer
  * cross entropy loss works well with a `sigmoid` last layer
  * if you're not sure, then `tanh` last layer, with squared loss, works well
* the softmax layer:
  * creates a probability distribution, ie a set of outputs, that sum to 1, and each lie in the range `0 <= x <= 1`
  * can create this probability distribution either across all output planes, with a boardsize of 1
    * this is the default
  * or else a per-plane probability distribution
    * add option `->perPlane()`

Train
-----

```c++
for( int epoch = 0; epoch < 12; epoch++ ) {
    float loss = net->epochMaker()
       ->learningRate(0.002)->batchSize(128)->numExamples(60000)
       ->inputData(mydata)
       ->labels(labels)
       ->runFromLabels( &trainNumRight );
    cout << "Loss L " << loss << " number correct: " << trainNumRight << endl;
}
```

Test
-------

```c++
net->setBatchSize(batchSize);
net->propagate(somenewdata);
float *results = net->getResults(); // to get results
float loss = net->calcLossFromLabels( labels ); // calc loss
int numberCorrect = net->calcNumRight( labels ); // check accuracy
```

Data format
===========

Input data should be provided in a contiguous array, of floats.  "group by" order should be:

* training example id
* input plane
* board row
* board column

Expected output data should be provided as a contiguous array, of floats. "group by" order should be:

* training example id
* output plane (eg, corresponds to filter id, for convolutional network)
* output row
* output column

Labels are simply an integer array, with one number, zero-based, per training example, or per test example.

Pre-requisites
==============

- git
- cmake
- gcc
- g++
- An OpenCL-compatible driver installed, and OpenCL-compatible GPU
  - tested using beignet, which provides OpenCL 1.2; and on CUDA 6.5 driver
- opencl-headers
- make 

To build
========

```bash
git clone --recursive https://github.com/hughperkins/ClConvolve.git
cd ClConvolve
mkdir build
cd build
cmake ..
make
```

Note:
* dont forget the `--recursive`, when you clone, else when you build it will complain about OpenCLHelper missing
* you might need to play around with commands such as `git submodule update` occasionally, to pull down new OpenCLHelper updates

Linking
=======

You will need:
- libClConvolve.so
- *.cl files

The *.cl files should be in the current working directory at the time that you call into any ClConvolve methods.

Correctness checking
====================

* For forward propagation:
  * We slot in some numbers, calculate the results manually, and compare with results actually obtained
  * We also forward propagate pictures/photos, and check the results look approximately like what we would expect
* For backward propagation:
  * We use numerical validation, since the sum of the square of the weight changes, divided by the learning rate, approximately equals the change in loss.  Or it should. We test this :-)
* Standard test sets
  * Checked using implementations for MNIST, and NORB is in progress

Unit-testing
============

Concepts
--------

* Network optimization is stochastic, and there are typically numerous local minima, into which the optimization can get stuck
* For unit testing, this is not very suitable, since unit tests must run repeatably, reliably, quickly
* Therefore, for unit-testing, the network weights are preset to a fixed set of values
  * using a random number generator with a fixed seed
  * or by explicitly giving a hard-coded set of weights
* Then, the test checks that the network converges to better than an expected loss, and accuracy, within a preset number of epochs
* We also have unit tests for forward-propagation, and backward propagation, as per section [Correctness checking](#correctness-checking) above.

Implementation
--------------

* Using googletest, which:
  * compiles quickly
  * gives awesome colored output
  * lets you choose which tests to run using `--gtest_filter=` option
* Dont need to install anything: it's included in the `thirdparty` directory, and added to the build automatically
* To run the unit tests:
```bash
make unittests
./unittests
```
* To run just the unittests for eg `testbackprop`, do:
```bash
make unittests
./unittests --gtest_filter=testbackprop.*
```
* To skip any slow tests, do:
```bash
./unittests --gtest_filter=-*SLOW*
```

Formulae notes
==============

These are generic formulae, but just putting them here, so I remember them ;-)

![formulae](notes-formulae.png)

Development notes
=================

- if you want to modify things, please feel free to fork this repository, tweak things, and send a pull request
- note that declarations in the header files are generated automatically.  [cogapp](http://nedbatchelder.com/code/cog/) generator provides
the framework, and [cog_addheaders.py](cog_addheaders.py) is a specific generator for header file declarations. You don't need this to
build the sources, but if you do have python installed, and you flip the `PYTHON_AVAILABLE` switch in the 
cmake configuration, then header file declarations will be updated for you automatically :-)

What's done / what's planned
============================

* Done:
  * forward/backward propagation, for convolutional networks, using OpenCL
  * square loss
  * zero-padding
  * relu activation
  * tanh activation
  * linear activation
  * some optimization of the OpenCL kernels
  * can save/load weights
  * can use 'fluent' style to setup the networks
  * unit-tests for forward propagation
  * numerical validation for backward propagation
* Planned, short-term:
  * ~~softmax activation function~~ done
  * ~~cross entropy loss~~ done
  * ~~multinomial cross entropy loss~~ done
  * get working with [kgs go data](https://github.com/hughperkins/kgsgo-dataset-preprocessor)
  * symmetric filters
  * maybe L2 regularization?
  * mpi so can run over several gpus, spread across multiple hosts???
    * implemented mpi in `testmnist-mpi`.  If works ok, will generalize to something more permanent
* Plausible, medium-term (pull requests welcome):
  * generalization to non-square images
  * generalization to larger images
  * drop-out
  * Python bindings?
  * ~~max-pooling?~~ done
  * read network from a config file?

Recent changes
==============

Dates are dates of code change / commit, rather than date merged into master, or tagged.
* 25th January:
  * Added gpu implementation for max-pooling forward-prop
  * Added padZeros option for max-pooling
* 24th January:
  * added max-pooling layer (albeit in cpu for now)
  * created draft 'lenet5' implementation, but it's not quite the same, specifically:
    * lenet-5 has RBF layers at the end
    * lenet-5 has multiple of these RBF and fully-connected layers at the end
    * lenet-5 is not using max-pooling but something more like average-pooling, and it has an activation function applied (sigmoid)
  * added the mnist training config from [convjs](http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)
  * noticed we are actually in January, not December, and updated the name of the month in this section appropriately :-P
* 23rd January:
  * created `testmnist-mpi`, to experiment with using mpi to parallelize across multiple compute nodes (which must each have a GPU, which GPUs must ideally each be the same model/specifications)
* 22nd January:
  * re-added FullyConnectedLayer, which is now a wrapper around ConvolutionalLayer, with one filter per output node.  So, if we want a 28x28 board as the output, this will need 784 filters in the underlying convolutional layer, which
sounds excessive, but this is how a fully connected layer works :-)
  * best mnist accuracy now at 98.6%
* 21st January:
  * added softmax layer, for per-column configuration, ie multi-planar output, with boardsize 1
    * tested once on mnist: 97.65% test accuracy after 12 epochs; 98.09% after 20 epochs
* week up to 21st January: 
  * added sigmoid activation
  * added cross-entropy loss layer
  * migrated to recurse on dLoss/dSum, rather than dLoss/dOutput, ie on partial derivative of loss with input to activation function for each neuron, rather than with output.  Recursing on input instead of output is faster
  * changed learning rate, so that the square of the sum of the weight changes equals approximately the change in loss, for smallish changes in w, so that we can numerically validate easily
  * validated backpropagation numerically
  * migrated to use explicit square-loss layer
  * moved sources to `src` sub-directory, so root directory cleaner
  * created `SLOW_` prefix for slow tests, so can run with `gtest_filter=-SLOW*` to ignore slow tests

Third-party libraries
=====================

* [OpenCLHelper](https://github.com/hughperkins/OpenCLHelper)
* [clew](https://github.com/martijnberger/clew)
* [libpng++](http://www.nongnu.org/pngpp/doc/0.2.1/)

License
=======

[Mozilla Public License 2.0](http://mozilla.org/MPL/2.0/)


