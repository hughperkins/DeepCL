ClConvolve
==========

OpenCL library to train deep convolutional networks
- C++
- OpenCL
- Deep convolutional

Target usage:
- 19 x 19 Go boards, eg something similar to [Clark and Storkey](http://arxiv.org/abs/1412.3409) or [Maddison, Huang, Sutskever and Silver](http://arxiv.org/abs/1412.6564)
- Also works on MNIST 28 x 28 boards (but you need to pad them to 29 x 29; the size of the board/image should be an odd number)

Neural Net API
==============

*Basic fully connected layers*

Example, with 1 fully connected layer:
```
#include "NeuralNet.h"

NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
net->fullyConnectedMaker()->planes(2)->boardSize(1)->insert();
net->print();
for( int epoch = 0; epoch < 100; epoch++ ) {
    net->epochMaker()
       ->learningRate(3)->batchSize(4)->numExamples(4)
       ->inputData(mydata)->expectedOutputs(myExpectedResults)
       ->run();
    cout << "Loss L " << net->calcLoss(expectedOutputs) << endl;
    AccuracyHelper::printAccuracy( numImages, numClasses, trainingLabels, net->getResults() );
}
delete net;
```

Notes:
* fully connected layers run on cpu for now

*Convolutional layers*

For convolutional layer, you can do:
```
net->ConvolutionalMaker()->numFilters(2)->filterSize(5)->insert();
```
Or:
```
net->ConvolutionalMaker()->numFilters(2)->filterSize(5)->padZeros()->insert();
```
Notes:
* convolutional layers forward-prop runs on gpu, via OpenCL
* back-prop is on cpu for now

Convolution layer options:
* add `->padZeros()` to pad the input board with zeros, so the output board and input board are the same size

*General*

Options for layer creation, for any layer type:
* add `->biased()` turn on bias
* add `->biased(1)` turn on bias (same as `->biased()`)
* add `->biased(0)` turn off bias (default)
* add `->linear()` choose linear activation
* add `->relu()` choose Relu activation
* add `->tanh()` choose Tanh activation (current default, but defaults can change...)

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
  - tested using beignet, which provides OpenCL 1.2, and CUDA 6.5 driver
- opencl-headers
- libpng++
- make 

To build
========

    git clone --recursive https://github.com/hughperkins/ClConvolve.git
    cd ClConvolve
    mkdir build
    cd build
    cmake ..
    make

Note:
* dont forget the `--recursive`, when you clone, else when you build it will complain about OpenCLHelper missing

Linking
=======

You will need:
- libClConvolve.so
- ClConvolve.cl

ClConvolve.cl should be in the current working directory at the time that you call into any clconvolve methods.

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

Sample/test
===========

*Neural net API*

* There are various samples in the [test](test) subdirectory.  The following are currently the most relevant:
  * `test2layerfullyconnected.cpp`  - use fully connected 2 layers, to learn and/or/xor
  * `testsimpleconvolve.cpp`      - check OpenCL convolve working, see how to use this
  * `testsimpleconvolvenet.cpp`   - use 1 layer convolutional network to learn toy 3x3 boards
  * `testneuralnetmnist.cpp`        - use 1 layer fully connected layer to learn MNIST
  * `testneuralnetmnistconvolve.cpp`  - learn mnist training images, using a single convolutional layer, then check against test set.  You can run it as follows, to get 82.3% test accuracy:
```
./testneuralnetmnistconvolve numtrain=10000 numtest=1000 batchsize=100 learningrate=0.1 biased=1 numepochs=20
```
... or you can experiment with the parameters.  Note that numtrain and numtest must be exact multiples of the batchsize

Third-party libraries
=====================

* [OpenCLHelper](https://github.com/hughperkins/OpenCLHelper)
* [clew](https://github.com/martijnberger/clew)
* [libpng++](http://www.nongnu.org/pngpp/doc/0.2.1/)

License
=======

[Mozilla Public License 2.0](http://mozilla.org/MPL/2.0/)


