#include "OpenCLHelper.h"
#include "ClConvolve.h"

#include <iostream>

#include "BoardHelper.h"
#include "MnistLoader.h"
#include "BoardPng.h"
#include "Timer.h"

using namespace std;

void run( int argc, char *argv[] ) {
    Timer timer;

    int boardSize;
    int N;
    int ***boards = MnistLoader::loadImages( "../data/mnist", "train", &N, &boardSize );
    timer.timeCheck("after load images");

    BoardPng::writeBoardsToPng( "testarraysquare-afterload.png", boards, min(N, 100), boardSize );

    int ***results = BoardsHelper::allocateBoards( N, boardSize );
    int filterSize;
    int **ofilter = BoardHelper::loadBoard( "../data/filt5-o.txt", &filterSize );
    int **vertfilter = BoardHelper::loadBoard( "../data/filt5-vert.txt", &filterSize );
    int **blurfilter = BoardHelper::loadBoard( "../data/filt5.txt", &filterSize );
    int **pointfilter = BoardHelper::loadBoard( "../data/filt5-point.txt", &filterSize );
    int ***filters = BoardsHelper::allocateBoards( 4, filterSize );
    BoardHelper::copyBoard( filters[0], ofilter, filterSize );
    BoardHelper::copyBoard( filters[1], vertfilter, filterSize );
    BoardHelper::copyBoard( filters[2], blurfilter, filterSize );
    BoardHelper::copyBoard( filters[3], pointfilter, filterSize );

    ClConvolve::convolveImage( boardSize, filterSize, &(boards[0][0][0]), &(ofilter[0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImage-ints.png", results, 1, boardSize );

    ClConvolve::convolveImages( N, boardSize, filterSize, &(boards[0][0][0]), &(ofilter[0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImages-ints.png", results, min(100,N), boardSize );

    ClConvolve::convolveImageCubes( N, 1, 1, boardSize, filterSize, &(boards[0][0][0]), &(ofilter[0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-1plane-1filter.png", results, min(100,N), boardSize );

    ClConvolve::convolveImageCubes( N / 4, 1, 4, boardSize, filterSize, &(boards[0][0][0]), &(filters[0][0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-1plane-4filter.png", results, min(100,N), boardSize );

    ClConvolve::convolveImageCubes( N / 4, 4, 1, boardSize, filterSize, &(boards[0][0][0]), &(filters[0][0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-4plane-1filter.png", results, min(100,N), boardSize );

    BoardsHelper::deleteBoards( &boards, N, boardSize );
    BoardsHelper::deleteBoards( &filters, N, boardSize );
    BoardsHelper::deleteBoards( &results, N, boardSize );
    BoardHelper::deleteBoard( &ofilter, filterSize );
    BoardHelper::deleteBoard( &vertfilter, filterSize );
    BoardHelper::deleteBoard( &blurfilter, filterSize );
    BoardHelper::deleteBoard( &pointfilter, filterSize );
}

int main( int argc, char *argv[] ) {
    run( argc, argv );
    return 0;
}


