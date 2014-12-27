#include "OpenCLHelper.h"
// #include "ClConvolve.h"

#include <iostream>

#include "BoardHelper.h"
#include "MnistLoader.h"
#include "BoardPng.h"
#include "Timer.h"
//#include "utils/memoryanalysis.cpp"

using namespace std;

void run( string mnistDir, int numPlanes, int numFilters, int numWorkgroupBoards ) {
    Timer timer;

    OpenCLHelper cl;
  
    int boardSize;
    int N = 60000;
    int ***boards = MnistLoader::loadImages( mnistDir, "train", &N, &boardSize );
    timer.timeCheck("after load images");

    BoardPng::writeBoardsToPng( "test4dconvolve1-1.png", boards, min(N, 100), boardSize );

    int ***results = BoardsHelper::allocateBoards( N, boardSize );
    int filterSize;
    int **filter = BoardHelper::loadBoard( "../data/filt5-o.txt", &filterSize );
    int **ofilter = BoardHelper::loadBoard( "../data/filt5-o.txt", &filterSize );
    int **vertfilter = BoardHelper::loadBoard( "../data/filt5-vert.txt", &filterSize );
    int **blur = BoardHelper::loadBoard( "../data/filt5.txt", &filterSize );
    int **point = BoardHelper::loadBoard( "../data/filt5-point.txt", &filterSize );

    // filters
    int ***filters = BoardsHelper::allocateBoards( 4, filterSize );
    BoardHelper::copyBoard( filters[0], blur, filterSize );
    BoardHelper::copyBoard( filters[1], ofilter, filterSize );
    BoardHelper::copyBoard( filters[2], vertfilter, filterSize );
    BoardHelper::copyBoard( filters[3], point, filterSize );
    float ***filtersFloat = BoardsHelper::allocateBoardsFloats( 4, filterSize );
    BoardsHelper::copyBoards( filtersFloat, filters, 4, filterSize );
    BoardsHelper::printBoards( filtersFloat, 4, filterSize );

    // boards
    float ***boardsFloat = BoardsHelper::allocateBoardsFloats( N, boardSize );
    BoardsHelper::copyBoards( boardsFloat, boards, N, boardSize );
    // normalize mean of each board
    for( int n = 0; n < N; n++ ) {
       float sum = 0;
       float count = 0;
       float thismax = 0;
       for( int i = 0; i < boardSize; i++ ) {
          for( int j = 0; j < boardSize; j++ ) {
              count++;
              sum += boardsFloat[n][i][j];
              thismax = max( thismax, boardsFloat[n][i][j] );
          }
       }
       float mean = sum / count;
//       cout << "mean " << mean << endl;
       for( int i = 0; i < boardSize; i++ ) {
          for( int j = 0; j < boardSize; j++ ) {
              boardsFloat[n][i][j] = boardsFloat[n][i][j] / thismax - 0.1;
          }
       }       
    }

    // results
    //float ***resultsFloat = BoardsHelper::allocateBoardsFloats( N, boardSize );
    float *resultsFloat = new float[N * boardSize * boardSize];

    timer.timeCheck("before convolve");
    CLKernel *kernel = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float" );
    timer.timeCheck("after compile");
    cout << "numplanes " << numPlanes << " numfilters " << numFilters << endl;
    kernel->in( numPlanes )->in( numFilters )->in(boardSize)->in(filterSize);
    kernel->input( N * boardSize * boardSize, &(boardsFloat[0][0][0]) );
    kernel->input( 4 * filterSize * filterSize, &(filtersFloat[0][0][0]) );
    kernel->output( N * boardSize * boardSize, resultsFloat );
    timer.timeCheck("after dataload");
    kernel->run_1d( numWorkgroupBoards * 1 * boardSize * boardSize, boardSize * boardSize / 2 );
    timer.timeCheck("after run");
    delete kernel;

    for( int i = 0; i < 20; i++ ) {
        cout << "resultsFloat[" << i << "] " << resultsFloat[i] << endl;
    }

    timer.timeCheck("after convolve");
    BoardPng::writeBoardsToPng( "test4dconvolve-floats-consolveimagecubes_float.png", resultsFloat, 16, boardSize );

    timer.timeCheck("before convolve no zeropadding");
    kernel = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float_nopadzeros" );
    timer.timeCheck("after compile no zeropadding");
    int outputBoardSize = ( boardSize - filterSize + 1 );
    cout << "numplanes " << numPlanes << " numfilters " << numFilters << endl;
    kernel->in( numPlanes )->in( numFilters )->in(boardSize)->in(filterSize);
    kernel->input( N * boardSize * boardSize, &(boardsFloat[0][0][0]) );
    kernel->input( 4 * filterSize * filterSize, &(filtersFloat[0][0][0]) );
    kernel->output( N * outputBoardSize * outputBoardSize, resultsFloat );
    timer.timeCheck("after dataload no zeropadding");
    kernel->run_1d( numWorkgroupBoards * 1 * outputBoardSize * outputBoardSize, outputBoardSize * outputBoardSize / 2 );
    timer.timeCheck("after run no zeropadding");
    delete kernel;

    for( int i = 0; i < 10; i++ ) {
        cout << "resultsFloat[" << i << "] " << resultsFloat[i] << endl;
    }

    timer.timeCheck("after convolve");
    BoardPng::writeBoardsToPng( "test4dconvolve-floats-consolveimagecubes_float_nopadzeros.png", resultsFloat, 16, boardSize - filterSize + 1 );
}

int main( int argc, char *argv[] ) {
    if( argc != 5 ) {
        cout << "Usage: " << argv[0] << " [mnist directory] [num planes] [num filters] [num workgroup boards]" << endl;
        exit(-1);
    }
    string mnistDir = argv[1];
    int numPlanes = atoi(argv[2]);
    int numFilters = atoi(argv[3]);
    int numWorkgroupBoards = atoi(argv[4]);
    run( mnistDir, numPlanes, numFilters, numWorkgroupBoards );
    return 0;
}


