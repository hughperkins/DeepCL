ClConvolve
==========

Current status: DRAFT, IN PROGRESS

Concept: OpenCL library to run convolutions on stacks of input images, using stacks of filters

Current status:
* Seems like convolution on its own isnt very self-contained, possibly, so building a more general
OpenCL neural net library... :-P

Neural Net API
==============

Example:

    #include "NeuralNet.h"
    
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->insert();
    net->print();
    for( int epoch = 0; epoch < 100; epoch++ ) {
        net->epochMaker()
           ->learningRate(3)->batchSize(4)->numExamples(4)
           ->inputData(ldc.data)->expectedOutputs(ldc.expectedResults)
           ->run();
        cout << "Loss L " << net->calcLoss(expectedOutputs) << endl;
        AccuracyHelper::printAccuracy( numImages, numClasses, trainingLabels, net->getResults() );
    }
    delete net;

Convolution API
===============

Simply call:

    ClConvolve::convolveImageCubes( int numImages, int numInputPlanes, int numFilters, int imageWidth, int filterWidth,
           int *images, int *filters, int *results );

Or for floats:

    ClConvolve::convolveImageCubes( int numImages, int numInputPlanes, int numFilters, int imageWidth, int filterWidth,
           float *images, float *filters, float *results );

- you need to provide the images, filters, and results arrays as contiguous arrays of integers
- the images array should consist of numImages cubes of images
  - each cube should have numInputPlanes planes
    - each plane should have imageWidth * imageWidth values, representing one plane of the input image
- the filters array should consist of numFilters * numInputPlanes images, each of filterWidth * filterWidth ints
- the output results array should be able to hold numImages * numFilters * imageWidth * imageWidth ints

Pre-requisites
==============

- git
- cmake
- gcc
- g++
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
    make ClConvolve

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

    int **BoardHelper::allocateBoard( int boardSize ); // allocates square array of size 
                                                       // boardSize by boardSize
    int ***BoardsHelper::allocateBoards( int numBoards, int boardSize ); 
                 // allocates numBoards square arrays of size 
                 // boardSize by boardSize
    BoardHelper::deleteBoard( int *p_board, int boardSize );
    BoardsHelper::deleteBoards( int *p_boards, int numBoards, int boardSize );

Use like this:
    int ***boards = BoardHelper::allocateBoards( 10, 19 ); // 10 boards, 19 by 19 each
    // &(boards[0][0]) is a contiguous array, of size 10 * 19 * 19
    // or you can use boards[boardIndex][row][col] to access as a 3d array

To clean up at the end:
    BoardHelper::deleteBoards( &boards, 10, 19 );

Also available for floats:

    float **BoardHelper::allocateBoardFloats( int boardSize ); // allocates square array of size 
                                                       // boardSize by boardSize
    int ***BoardsHelper::allocateBoardsFloats( int numBoards, int boardSize ); 
                 // allocates numBoards square arrays of size 
                 // boardSize by boardSize
    BoardHelper::deleteBoard( float *p_board, int boardSize );
    BoardsHelper::deleteBoards( float *p_boards, int numBoards, int boardSize );

*MnistLoader*

- use to load the mnist data set, that LeCun has provided
- download the mnist data files into folder ../data/mnist, from LeCun's download page
- use like this:

    int boardSize;
    int N;
    int ***boards = MnistLoader::loadImages( "../data/mnist", "train", &N, &boardSize );

- `boards` now contains `N` boards, of `boardSize` by `boardSize`, loaded from the `"train"` dataset
- if you want the 'test' dataset, pass in `"t10k"` instead of `"train"`
- you can put the datafiles in a different folder, just pass in the path to that folder as the first argument

*BoardPng*

* use to print one or more boards to a png file
* use like this:

    BoardPng::writeBoardsToPng( "testarraysquare-afterload.png", boards, min(N, 100), boardSize );

* where:
** first argument is filename to write png to
** second is array of boards, in same format as returned by BoardsHelper::allocateBoards
** third argument is number of boards to write to png
** third argument is the length of one side of each board

Sample/test
===========

*Neural net API*

* Run:

    ./test2layerfullyconnected

If you open the test2layerfullyconnected.cpp file, you can see how it works, and in the `main` method, you can choose
different submethods you can call.

*Convolutions API*

* Run:

    ./testarraysquare [directory containing mnist datafiles]

* You should see various png files appear in the current directory
* Examples of some of the samples in testarraysquare:

    ClConvolve::convolveImage( boardSize, filterSize, &(boards[0][0][0]), 
        &(ofilter[0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImage-ints.png", results, 
        1, boardSize );
    ClConvolve::convolveImages( N, boardSize, filterSize, &(boards[0][0][0]), 
        &(ofilter[0][0]), 
        &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImages-ints.png", results, 
        min(100,N), boardSize );
    ClConvolve::convolveImageCubes( N, 1, 1, boardSize, filterSize, 
        &(boards[0][0][0]), &(ofilter[0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-1plane-1filter.png", 
        results, min(100,N), boardSize );
    ClConvolve::convolveImageCubes( N / 4, 1, 4, boardSize, filterSize, 
        &(boards[0][0][0]), &(filters[0][0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-1plane-4filter.png", 
        results, min(100,N), boardSize );
    ClConvolve::convolveImageCubes( N / 4, 4, 1, boardSize, filterSize, 
        &(boards[0][0][0]), &(filters[0][0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-4plane-1filter.png", 
        results, min(100,N), boardSize );

Third-party libraries
=====================

* [OpenCLHelper](https://github.com/hughperkins/OpenCLHelper)
* [clew](https://github.com/martijnberger/clew)
* libpng++

License
=======

[Mozilla Public License 2.0](http://mozilla.org/MPL/2.0/)


