<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [ClConvolve](#clconvolve)
- [Neural Net API](#neural-net-api)
  - [Create a net](#create-a-net)
  - [Add some layers](#add-some-layers)
  - [Train](#train)
  - [Predict](#predict)
- [Data format](#data-format)
- [Pre-requisites](#pre-requisites)
- [To build](#to-build)
- [Linking](#linking)
- [Sample/test](#sampletest)
- [Unit-testing](#unit-testing)
  - [Concepts](#concepts)
  - [Unit-testing implementation](#unit-testing-implementation)
- [Formulae notes](#formulae-notes)
- [Development notes](#development-notes)
- [Helper methods](#helper-methods)
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

* Please use a Convolutional Layer, and set the filter size to be identical to the output
size of the previous layer.

*Loss layer*

Please add one loss layer.  You can choose between squared loss and cross-entropy loss:
```c++
net->squareLossMaker()->insert();
net->crossEntropyLossMaker()->insert();
```
* squared loss works well with a `tanh` last layer
* cross entropy loss works well with a `sigmoid` last layer
* if you're not sure, then `tanh` last layer, with squared loss, works well

New: can also choose softmax layer, if your second to last layer has a boardsize of 1:
```c++
net->softMaxLossLayer()->insert();
```

Train
-----

```c++
for( int epoch = 0; epoch < 12; epoch++ ) {
    net->epochMaker()
       ->learningRate(0.1)->batchSize(128)->numExamples(60000)
       ->inputData(mydata)->expectedOutputs(myExpectedResults)
       ->run();
    cout << "Loss L " << net->calcLoss(expectedOutputs) << endl;
    AccuracyHelper::printAccuracy( numImages, numClasses, trainingLabels, net->getResults() );
}
```

Predict
-------

```c++
net->setBatchSize(batchSize);
net->propagate(somenewdata);
float *results = net->getResults();
```

Data format
===========

Input data should be provided in a contiguous array.  "group by" order should be:

* training example id
* input plane
* board row
* board column

Expected output data should be provided as a contiguous array. "group by" order should be:

* training example id
* output plane (eg, corresponds to filter id, for convolutional network)
* output row
* output column

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

Sample/test
===========

How to check for correctness, and speed?  Perhaps the easiest way is to train against MNIST
* First you need to download MNIST.  The files need to be placed in the `data\mnist` directory.  If you're on linux, and you're currently in the `build` subdirectory, you could do:
```bash
cd ../data
mkdir mnist
cd mnist
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ../../build
```
* Then, you can train against MNIST using a net created eg as follows:
```c++
NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(28)->instance();
net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
net->convolutionalMaker()->numFilters(10)->filterSize(24)->linear()->biased()->insert();
net->softMaxLossMaker()->insert();
net->setBatchSize(128);
```
* Network details:
  * First line creates a NeuralNet object, together with a first `InputLayer` layer, to receive the incoming data
    * When we create the net, we specify the size of the incoming data, ie number of planes per example, and size of each plane
  * The second line creates a a convolutional layer with 32 feature maps, each with a filter size of 5.  Non-linearity is relu
  * The next line looks like a convolutional layer, and in fact it is, but it's being used as a fully-connected layer, since the filter size is identical to the output of the previous layer, and padZeros is not enabled
    * Each filter will result in one single output node, for a total of 10 output nodes, dimensioned as 10 planes, each of a 1x1 board
  * Finally, we add a SoftMaxLoss layer, which handles:
    * generating appropriate loss signals to drive the network
    * receive our labels array
    * calculate loss
    * calculate number correct
  * You need to set the batch size before passing in any input data, since this sets up the internal buffer sizes.  Failure to do this will result in seg faults and other nasty errors :-P
* There is an implementation of this network, including loading mnist, and normalizing it, at [testmnist-softmax.cpp](test/testmnist-softmax.cpp)
  * You can build and run it as follows:
```bash
make testmnist-softmax
./testmnist-softmax numfilters=32
```
* Here are some results I obtained, using an Amazon AWS GPU instance, which has an NVidia GRID K520 GPU:

|ClConvolve version| Learning rate | Number filters | Filter size | Number filter layers | Number epochs | Epoch time | Test accuracy |
|-------|----|---------------|--------------|----------------------|----------------|---------------|-----------|
|v0.1 (*) |0.1  |32             | 5            | 1                    | 12          | 18.2 seconds      |97.3+/-0.2% |
|v0.1 (*) |0.02  |32             | 5            | 2                    | 50          | 101 seconds      |98.2+/-0.3% |
|~v0.2c (*) |0.0001  |32             | 5            | 1                    | 12          | 17.2 seconds      | 97.3 +/- ? |
| ~v0.2c (*)  | 0.0001 | 32 | 5 | 2 | 12 | 80.4 seconds | 98.2 +/- ? |
| next v0.3 | 0.0001 | 32 | 5 | 1 | 12 | 17.2 seconds | 97.5 +/- ? |
| next v0.3 | 0.0001 | 32 | 5 | 1 | 20 | 17.2 seconds | 98.1 +/- ? |
| next v0.3 | 0.0001 | 32 | 5 | 2 | 12 | 80.5 seconds | 98.3 +/- ? |

* (*) Using earlier `testneuralnetmninstconvolve-experimental` executable, which used a `tanh` last layer activation, square loss, and provided an expected values array of `-0.5` for `false`, and `+0.5` for `true

Unit-testing
============

Concepts
--------

* Unit-testing is a challenge, with neural nets, because:
  * training is stochastic
  * there are many local minima
* So, if you run a net once, you might get a MSE loss of 0.000001, and a perfect accuracy, but if you run it 10 times, 
it might fail horribly on one run, with a much higher MSE, and a non-perfect score.  Even for toy problems, like 'and'
or 'or' gates
* One could run each test a hundred times, and check that at least 90% of them pass, but this would be slow, and would
still fail sometimes
* One could run many times, and check that at least once, there is perfect accuracy, but this doesnt show that learning
is correct, could just be by random chance
* I think that what we really want to show is not that we always reach the global minimum, but that:
  * the network is correctly stable in the global minimum, and
  * if we move the network away from the global minimum slightly, it will converge, correctly, on this global minimum
* Therefore, the current plan is to run each network once or twice, till it finds a global minimum, then record the
weights from 15-20 iterations higher up, which we've seen converge on the global solution
* Then, for our unit tests, we simply re-use these same weights, and check that the loss after 15-20 iterations is
better than for example 0.0001 (for toy problems)
* Result: repeatable, fast, unit tests

Unit-testing implementation
---------------------------

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

Helper methods
==============

*Contiguous arrays*

To allocate contiguous 2d and 3d arrays:
```c++
int **BoardHelper::allocateBoard( int boardSize ); // allocates square array of size 
                                                   // boardSize by boardSize
int ***BoardsHelper::allocateBoards( int numBoards, int boardSize ); 
             // allocates numBoards square arrays of size 
             // boardSize by boardSize
BoardHelper::deleteBoard( int *p_board, int boardSize );
BoardsHelper::deleteBoards( int *p_boards, int numBoards, int boardSize );
```
Use like this:
```c++
int ***boards = BoardHelper::allocateBoards( 10, 19 ); // 10 boards, 19 by 19 each
// &(boards[0][0]) is a contiguous array, of size 10 * 19 * 19
// or you can use boards[boardIndex][row][col] to access as a 3d array
```
To clean up at the end:
```c++
BoardHelper::deleteBoards( &boards, 10, 19 );
```
Also available for floats:
```c++
float **BoardHelper::allocateBoardFloats( int boardSize ); // allocates square array of size 
                                                   // boardSize by boardSize
int ***BoardsHelper::allocateBoardsFloats( int numBoards, int boardSize ); 
             // allocates numBoards square arrays of size 
             // boardSize by boardSize
BoardHelper::deleteBoard( float *p_board, int boardSize );
BoardsHelper::deleteBoards( float *p_boards, int numBoards, int boardSize );
```
*MnistLoader*

- use to load the mnist data set, that LeCun has provided
- download the mnist data files into folder ../data/mnist, from LeCun's download page
- use like this:
```c++
int boardSize;
int N;
int ***boards = MnistLoader::loadImages( "../data/mnist", "train", &N, &boardSize );
int *labels = MnistLoader::loadLabels( "../data/mnist", "train", &N );
```
- `boards` now contains `N` boards, of `boardSize` by `boardSize`, loaded from the `"train"` dataset
- if you want the 'test' dataset, pass in `"t10k"` instead of `"train"`
- you can put the datafiles in a different folder, just pass in the path to that folder as the first argument

*BoardPng*

* use to print one or more boards to a png file
* use like this:
```c++
BoardPng::writeBoardsToPng( "testarraysquare-afterload.png", boards, min(N, 100), boardSize );
```
* where:
  * first argument is filename to write png to
  * second is array of boards, in same format as returned by BoardsHelper::allocateBoards
  * third argument is number of boards to write to png
  * third argument is the length of one side of each board

What's done / what's planned
============================

* Done:
  * forward/backward propagation, for convolutional networks, using OpenCL
  * square loss
  * zero-padding
  * relu activation
  * tanh activation
  * linear activation
  * some optimization of the OpenCL kernels, targeting 19x19 Go boards
  * can save/load weights
  * can use 'fluent' style to setup the networks
  * unit-tests for forward propagation
  * numerical validation for backward propagation
* Planned, short-term:
  * -softmax activation function- done
  * -cross entropy loss- done
  * -multinomial cross entropy loss- done
  * get working with [kgs go data](https://github.com/hughperkins/kgsgo-dataset-preprocessor)
  * symmetric filters
  * maybe L2 regularization?
* Plausible, medium-term (pull requests welcome):
  * generalization to non-square images
  * generalization to larger images
  * drop-out
  * Python bindings?

Recent changes
==============

Dates are dates of code change / commit, rather than date merged into master, or tagged.

* 21st December:
  * added softmax layer, for per-column configuration, ie multi-planar output, with boardsize 1
    * tested once on mnist: 97.65% test accuracy after 12 epochs; 98.09% after 20 epochs
* week up to 21st December: 
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


