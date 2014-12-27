#include "OpenCLHelper.h"
// #include "ClConvolve.h"

#include <iostream>
using namespace std;

#include "BoardHelper.h"
#include "MnistLoader.h"
#include "BoardPng.h"
#include "utils/Timer.h"
//#include "utils/memoryanalysis.cpp"

void run( int numPlanes, int numFilters, int numWorkgroupBoards ) {
    Timer timer;

    OpenCLHelper cl;
  
    int boardSize;
    int N = 60000;
    int ***boards = MnistLoader::loadImages( "/norep/Downloads/data/mnist", "train", &N, &boardSize );
    timer.timeCheck("after load images");

    BoardPng::writeBoardsToPng( "test4dconvolve1-1.png", boards, min(N, 100), boardSize );

    int ***results = BoardsHelper::allocateBoards( N, boardSize );
    int filterSize;
    int **filter;
    int piecesPlaced;
    BoardHelper::loadBoard( &filter, "../data/filt5-o.txt", &filterSize, &piecesPlaced );
    int **ofilter;
    BoardHelper::loadBoard( &ofilter, "../data/filt5-o.txt", &filterSize, &piecesPlaced );
    int **vertfilter;
    BoardHelper::loadBoard( &vertfilter, "../data/filt5-vert.txt", &filterSize, &piecesPlaced );
    int **blur;
    BoardHelper::loadBoard( &blur, "../data/filt5.txt", &filterSize, &piecesPlaced );
    int **point;
    BoardHelper::loadBoard( &point, "../data/filt5-point.txt", &filterSize, &piecesPlaced );

    // filters
    int ***filters = BoardsHelper::allocateBoards( 4, filterSize );
    BoardHelper::copyBoard( ofilter, filters[0], filterSize );
    BoardHelper::copyBoard( blur, filters[1], filterSize );
    BoardHelper::copyBoard( vertfilter, filters[2], filterSize );
    BoardHelper::copyBoard( point, filters[3], filterSize );
    float ***filtersFloat = BoardsHelper::allocateBoardsFloat( 4, filterSize );
    BoardsHelper::copyBoards( filtersFloat, filters, 4, filterSize );

    // boards
    float ***boardsFloat = BoardsHelper::allocateBoardsFloat( N, boardSize );
    BoardsHelper::copyBoards( boardsFloat, boards, N, boardSize );

    // results
    float ***resultsFloat = BoardsHelper::allocateBoardsFloat( N, boardSize );

//convolve_imagecubes_float( const int numInputPlanes, const int numFilters, 
//      const int boardSize, const int filterSize,
//      global const float *images, global const float *filters, global float *results )
    timer.timeCheck("before convolve");
    CLKernel *kernel = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float" );
    timer.timeCheck("after compile");
    cout << "numplanes " << numPlanes << " numfilters " << numFilters << endl;
    kernel->in( numPlanes )->in( numFilters )->in(boardSize)->in(filterSize);
    kernel->input( N * boardSize * boardSize, &(boardsFloat[0][0][0]) );
    kernel->input( 4 * filterSize * filterSize, &(filtersFloat[0][0][0]) );
    kernel->output( N * boardSize * boardSize, &(resultsFloat[0][0][0]) );
    timer.timeCheck("after dataload");
    kernel->run_1d( numWorkgroupBoards * 1 * boardSize * boardSize, boardSize * boardSize / 2 );
    timer.timeCheck("after run");

    for( int i = 0; i < 4; i++ ) {
        cout << "resultsFloat[" << i << "] " << resultsFloat[0][0][i] << endl;
    }

//    delete kernel;
    timer.timeCheck("after convolve");
    BoardPng::writeBoardsToPng( "testarraysquare-floats-consolveimagecubes_float.png", resultsFloat, 16, boardSize );
}

int main( int argc, char *argv[] ) {
    int numPlanes = atoi(argv[1]);
    int numFilters = atoi(argv[2]);
    int numWorkgroupBoards = atoi(argv[3]);
    run( numPlanes, numFilters, numWorkgroupBoards );
    return 0;
}


