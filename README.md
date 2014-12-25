ClConvolve
==========

Current status: DRAFT, IN PROGRESS

OpenCL library to run convolutions on stacks of input images, using stacks of filters

API
===

Simply call:

    ClConvolve::convolveImageCubes( int numImages, int numInputPlanes, int numFilters, int imageWidth, int filterWidth,
           int *images, int *filters, int *results );

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

