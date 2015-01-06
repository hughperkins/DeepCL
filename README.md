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

```
#include "ClConvolve.h"

NeuralNet *net = NeuralNet::maker()->planes(10)->boardSize(19)->instance();
```

* You can change the number of input planes, and the board size.

Add some layers
---------------

*Convolutional layers*

Eg:
```
net->ConvolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
```

* You can change the number of filters, and their size.  If you want, you can use any of the following options:
  * `->padZeros()`: pad the input board with zeros, so the output board is same size as the input
  * `->biased()` turn on bias
  * `->biased(1)` same as `->biased()`
  * `->biased(0)` turn off bias (default)
  * `->linear()` choose linear activation
  * `->relu()` choose Relu activation
  * `->tanh()` choose Tanh activation (current default, but defaults can change...)
* convolutional layers forward-prop and backward-prop both run on GPU, via OpenCL

*Fully connected layers*

eg:
```
net->fullyConnectedMaker()->planes(10)->boardSize(1)->insert();
```

* fully connected layers run on cpu for now
* For now, recommend to use a Convolutional Layer, and set the filter size to be identical to the output
size of the previous layer, and without `padZeros` activated.

Train the net
-------------

```
for( int epoch = 0; epoch < 12; epoch++ ) {
    net->epochMaker()
       ->learningRate(0.1)->batchSize(128)->numExamples(60000)
       ->inputData(mydata)->expectedOutputs(myExpectedResults)
       ->run();
    cout << "Loss L " << net->calcLoss(expectedOutputs) << endl;
    AccuracyHelper::printAccuracy( numImages, numClasses, trainingLabels, net->getResults() );
}
```

Print the net
-------------

You can print the network like this:
```
net->print();
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

```
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
- ClConvolve.cl

ClConvolve.cl should be in the current working directory at the time that you call into any clconvolve methods.

Sample/test
===========

How to check for correctness, and speed?  Perhaps the easiest way is to train against MNIST
* First you need to download MNIST.  The files need to be placed in the `data\mnist` directory.  If you're on linux, and you're currently in the `build` subdirectory, you could do:
```
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
* Then, you can train against MNIST using a 2-layer net created as follows:
```
NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(boardSize)->instance();
net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
net->convolutionalMaker()->numFilters(10)->filterSize(boardSize-4)->tanh()->biased(config.biased)->insert();
```
* Network details:
  * The first layer is a convolutional layer with 32 feature maps, each with a filter size of 5.  Non-linearity is using Relu
  * The second layer looks like a convolutional layer, and in fact it is, but it's being used as a fully-connected layer, since the filter size is identical to the output of the previous layer, and padZeros is not enabled
* There is an implementation of this network, including loading mnist, and normalizing it, at [testneuralnetmnistconvolve-experimental.cpp](https://github.com/hughperkins/ClConvolve/blob/master/test/testneuralnetmnistconvolve-experimental.cpp)
  * You can build and run it as follows:
```
make testneuralnetconvolve-experimental
./testneuralnetconvolve-experimental numfilters=32
```
* For me, after 12 epochs, the test accuracy was 97.3% +/-0.2%, which seems somewhat plausible, compared to other implementations [MNIST database](http://yann.lecun.com/exdb/mnist/)
  * Each epoch took 17 seconds, on an Amazon AWS GPU instance, which has an NVidia GRID K520 GPU, for 3.5 minutes total training.

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
```
make unittests
./unittests
```
* To run just the unittests for eg `testbackprop`, do:
```
make unittests
./unittests --gtest_filter=testbackprop.*
```

Development notes
=================

- if you want to modify things, please feel free to fork this repository, tweak things, and send a pull request
- note that the headers are generated using [cogapp](http://nedbatchelder.com/code/cog/)  . You don't need this to
build the sources, but if you do have python installed, and you flip the `PYTHON_AVAILABLE` switch in the 
cmake configuration, then header file declarations will be generated for you automatically when you build :-)

Helper methods
==============

*Contiguous arrays*

To allocate contiguous 2d and 3d arrays:
```
int **BoardHelper::allocateBoard( int boardSize ); // allocates square array of size 
                                                   // boardSize by boardSize
int ***BoardsHelper::allocateBoards( int numBoards, int boardSize ); 
             // allocates numBoards square arrays of size 
             // boardSize by boardSize
BoardHelper::deleteBoard( int *p_board, int boardSize );
BoardsHelper::deleteBoards( int *p_boards, int numBoards, int boardSize );
```
Use like this:
```
int ***boards = BoardHelper::allocateBoards( 10, 19 ); // 10 boards, 19 by 19 each
// &(boards[0][0]) is a contiguous array, of size 10 * 19 * 19
// or you can use boards[boardIndex][row][col] to access as a 3d array
```
To clean up at the end:
```
BoardHelper::deleteBoards( &boards, 10, 19 );
```
Also available for floats:
```
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
```
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
```
BoardPng::writeBoardsToPng( "testarraysquare-afterload.png", boards, min(N, 100), boardSize );
```
* where:
  * first argument is filename to write png to
  * second is array of boards, in same format as returned by BoardsHelper::allocateBoards
  * third argument is number of boards to write to png
  * third argument is the length of one side of each board

Third-party libraries
=====================

* [OpenCLHelper](https://github.com/hughperkins/OpenCLHelper)
* [clew](https://github.com/martijnberger/clew)
* [libpng++](http://www.nongnu.org/pngpp/doc/0.2.1/)

License
=======

[Mozilla Public License 2.0](http://mozilla.org/MPL/2.0/)


