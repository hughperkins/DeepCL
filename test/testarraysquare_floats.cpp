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

    if( argc != 2 ) {
        cout << "Usage: " << argv[0] << " [mnist data directory]" << endl;
        return;
    }
    string mnistDir = argv[1];
    cout << "Using mnist data from " << mnistDir << endl;

    int boardSize;
    int N;
    int ***boards = MnistLoader::loadImages( mnistDir, "train", &N, &boardSize );
    timer.timeCheck("after load images");

    BoardPng::writeBoardsToPng( "testarraysquare-afterload.png", boards, min(N, 100), boardSize );

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

    float ***boardsFloat = BoardsHelper::allocateBoardsFloats( N, boardSize );
    float ***filtersFloat = BoardsHelper::allocateBoardsFloats( 4, filterSize );
    float ***resultsFloat = BoardsHelper::allocateBoardsFloats( N, boardSize );

    BoardsHelper::copyBoards( boardsFloat, boards, N, boardSize );
    BoardsHelper::copyBoards( filtersFloat, filters, 4, filterSize );

    ClConvolve::convolveImageCubes( 1, 1, 1, boardSize, filterSize, &(boardsFloat[0][0][0]), &(filtersFloat[0][0][0]), &(resultsFloat[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImage-floats-1.png", resultsFloat, 1, boardSize );

    ClConvolve::convolveImageCubes( N, 1, 1, boardSize, filterSize, &(boardsFloat[0][0][0]), &(filtersFloat[0][0][0]), &(resultsFloat[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImages-floats-100.png", resultsFloat, min(100,N), boardSize );

    ClConvolve::convolveImageCubes( N, 1, 1, boardSize, filterSize, &(boardsFloat[0][0][0]), &(filtersFloat[0][0][0]), &(resultsFloat[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-1plane-1filter-floats.png", resultsFloat, min(100,N), boardSize );

    ClConvolve::convolveImageCubes( N / 4, 1, 4, boardSize, filterSize, &(boardsFloat[0][0][0]), &(filtersFloat[0][0][0]), &(resultsFloat[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-1plane-4filter-floats.png", resultsFloat, min(100,N), boardSize );

    ClConvolve::convolveImageCubes( N / 4, 4, 1, boardSize, filterSize, &(boardsFloat[0][0][0]), &(filtersFloat[0][0][0]), &(resultsFloat[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-4plane-1filter-floats.png", resultsFloat, min(100,N), boardSize );

    BoardsHelper::deleteBoards( &boardsFloat, N, boardSize );
    BoardsHelper::deleteBoards( &filtersFloat, 4, boardSize );
    BoardsHelper::deleteBoards( &resultsFloat, N, boardSize );

    BoardsHelper::deleteBoards( &boards, N, boardSize );
    BoardsHelper::deleteBoards( &filters, 4, boardSize );

    BoardHelper::deleteBoard( &ofilter, filterSize );
    BoardHelper::deleteBoard( &vertfilter, filterSize );
    BoardHelper::deleteBoard( &blurfilter, filterSize );
    BoardHelper::deleteBoard( &pointfilter, filterSize );
}

int main( int argc, char *argv[] ) {
    run( argc, argv );
    return 0;
}


