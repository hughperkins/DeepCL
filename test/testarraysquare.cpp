#include "OpenCLHelper.h"
#include "ClConvolve.h"

#include <iostream>
using namespace std;

#include "BoardHelper.h"
#include "MnistLoader.h"
#include "BoardPng.h"
#include "Timer.h"
//#include "utils/memoryanalysis.cpp"

void run( int argc, char *argv[] ) {
    float a[] = {1,3,5,7,11,15};

    Timer timer;

    OpenCLHelper cl(0);
    CLKernel *kernel = cl.buildKernel( "../test/testarraysquare.cl", "dp_square_float" );
    kernel->input( 6, a );
    float result[6];
    kernel->output( 6, result );
    size_t global = 6;
    size_t local = 6;
    kernel->run_1d( 6, 6 );
 
    for( int i = 0; i < 6; i++ ) {
        cout << "i " << i << " in[i] " << a[i] << " result[i] " << result[i] << endl;
    }

    delete kernel;

    int boardSize;
    int N = 60000;
    int workgroupsize = 128;
    int ***boards = MnistLoader::loadImages( "../data/mnist", "train", N, &boardSize );
    timer.timeCheck("after load images");


    BoardPng::writeBoardsToPng( "testarraysquare-1.png", boards, min(N, 100), boardSize );

//    float ***floatBoards = BoardsHelper::allocateFloatBoards( N, boardSize );
//    int *images_1d = new int[boardSize*boardSize*N];
    cout << "linear size " << (boardSize * boardSize * N) << endl;
    int ***results = BoardsHelper::allocateBoards( N, boardSize );

//    kernel = cl.buildKernel( "../test/testarraysquare.cl", "dp_square_int" );
//    kernel->input( 1, &boardSize );
//    kernel->input( N * boardSize * boardSize, &(boards[0][0][0]) );
//    kernel->output( N * boardSize * boardSize, &(results[0][0][0]) );
//    kernel->run_1d( N * boardSize * boardSize, workgroupsize );
//    delete kernel;

//    BoardPng::writeBoardsToPng( "testarraysquare-2.png", results, min(N, 100), boardSize );

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

//    ClConvolve::convolveImage( boardSize, filterSize, &(boards[0][0][0]), &(filter[0][0]), &(results[0][0][0]) );
//    BoardPng::writeBoardsToPng( "ClConvolve-convolveImage-ints.png", results, 1, boardSize );

//    ClConvolve::convolveImages( N, boardSize, filterSize, &(boards[0][0][0]), &(filter[0][0]), &(results[0][0][0]) );
//    BoardPng::writeBoardsToPng( "ClConvolve-convolveImages-ints.png", results, min(100,N), boardSize );

    ClConvolve::convolveImageCubes( N, 1, 1, boardSize, filterSize, &(boards[0][0][0]), &(filter[0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-1plane-1filter.png", results, min(100,N), boardSize );

    int ***filters = BoardsHelper::allocateBoards( 10, filterSize );
    BoardHelper::copyBoard( ofilter, filters[0], filterSize );
    BoardHelper::copyBoard( vertfilter, filters[1], filterSize );
    BoardHelper::copyBoard( blur, filters[2], filterSize );
    BoardHelper::copyBoard( point, filters[3], filterSize );

    ClConvolve::convolveImageCubes( N / 4, 1, 4, boardSize, filterSize, &(boards[0][0][0]), &(filters[0][0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-1plane-4filter.png", results, min(100,N), boardSize );

    ClConvolve::convolveImageCubes( N / 4, 4, 1, boardSize, filterSize, &(boards[0][0][0]), &(filters[0][0][0]), &(results[0][0][0]) );
    BoardPng::writeBoardsToPng( "ClConvolve-convolveImageCubes-4plane-1filter.png", results, min(100,N), boardSize );

    return;

    cout << "N " << N << " imagesize " << boardSize << " filtersize " << filterSize << endl;

    timer.timeCheck("before convolve");
    kernel = cl.buildKernel( "../test/testarraysquare.cl", "convolve_ints" );
    timer.timeCheck("after compile");
    kernel->input( 1, &boardSize );
    kernel->input( 1, &filterSize );
    kernel->input( N * boardSize * boardSize, &(boards[0][0][0]) );
    kernel->input( filterSize * filterSize, &(filter[0][0]) );
    kernel->output( N * boardSize * boardSize, &(results[0][0][0]) );
    timer.timeCheck("after dataload");
    kernel->run_1d( N * boardSize * boardSize, workgroupsize );
    timer.timeCheck("after run");
//    delete kernel;
    timer.timeCheck("after convolve");

    BoardPng::writeBoardsToPng( "testarraysquare-3.png", results, min(N, 100), boardSize );
    timer.timeCheck("after write to png");

    BoardHelper::deleteBoard( &filter, filterSize );
    BoardHelper::loadBoard( &filter, "../data/filt5-vert.txt", &filterSize, &piecesPlaced );
    timer.timeCheck("after compile");
    kernel->input( 1, &boardSize );
    kernel->input( 1, &filterSize );
    kernel->input( N * boardSize * boardSize, &(boards[0][0][0]) );
    kernel->input( filterSize * filterSize, &(filter[0][0]) );
    kernel->output( N * boardSize * boardSize, &(results[0][0][0]) );
    timer.timeCheck("after dataload");
    kernel->run_1d( N * boardSize * boardSize, workgroupsize );
    timer.timeCheck("after run");
    //delete kernel;
    timer.timeCheck("after convolve");

    BoardPng::writeBoardsToPng( "testarraysquare-4.png", results, min(N, 100), boardSize );
    timer.timeCheck("after write to png");

    BoardHelper::deleteBoard( &filter, filterSize );
    BoardHelper::loadBoard( &filter, "../data/filt5-o.txt", &filterSize, &piecesPlaced );

    // CLIntWrapper
 
    CLIntWrapper *imagesBuffer = cl.intWrapper( N * boardSize * boardSize, &(boards[0][0][0]) );
    CLIntWrapper *filterBuffer = cl.intWrapper( filterSize * filterSize, &(filter[0][0]) );
    CLIntWrapper *resultsBuffer = cl.intWrapper( N * boardSize * boardSize, &(results[0][0][0]) );
    imagesBuffer->copyToDevice();
    filterBuffer->copyToDevice();
    kernel->input( 1, &boardSize );     
    kernel->input( 1, &filterSize );
    kernel->input( imagesBuffer );
    kernel->input( filterBuffer );     
    kernel->output( resultsBuffer );
    timer.timeCheck("after dataload");
    kernel->run_1d( N * boardSize * boardSize, workgroupsize );
    timer.timeCheck("after convolve");
    delete kernel;
    
    timer.timeCheck("after copy to host");
    BoardPng::writeBoardsToPng( "testarraysquare-5.png", results, min(N, 100), boardSize );
    timer.timeCheck("after write to png");

    // floats

    float ***boardsFloat = BoardsHelper::allocateBoardsFloat( N, boardSize );
    BoardsHelper::copyBoards( boardsFloat, boards, N, boardSize );
    float ***resultsFloat = BoardsHelper::allocateBoardsFloat( N, boardSize );
    float **filterFloat = BoardHelper::allocateFloats( boardSize );
    int filter1dsize = boardSize * boardSize;
    int *filter1d = &(filter[0][0]);
    float *filterFloats1d = &(filterFloat[0][0]);
    for( int i = 0; i < filter1dsize; i++ ) {
       filterFloats1d[i] = filter1d[i];
    }

    timer.timeCheck("before convolve");
    kernel = cl.buildKernel( "../test/testarraysquare.cl", "convolve_floats" );
    timer.timeCheck("after compile");
    kernel->input( 1, &boardSize );
    kernel->input( 1, &filterSize );
    kernel->input( N * boardSize * boardSize, &(boardsFloat[0][0][0]) );
    kernel->input( filterSize * filterSize, &(filterFloat[0][0]) );
    kernel->output( N * boardSize * boardSize, &(resultsFloat[0][0][0]) );
    timer.timeCheck("after dataload");
    kernel->run_1d( N * boardSize * boardSize, workgroupsize );
    timer.timeCheck("after run");
    delete kernel;
    timer.timeCheck("after convolve");

    BoardPng::writeBoardsToPng( "testarraysquare-5.png", resultsFloat, min(N, 100), boardSize );
    timer.timeCheck("after write to png");


    BoardHelper::deleteBoard( &filter, filterSize );

//    BoardHelper::loadBoard( &filter, "../data/filt5-vert.txt", &filterSize, &piecesPlaced );

//    timer.timeCheck("before convolve");
//    kernel = cl.buildKernel( "../test/testarraysquare.cl", "convolve_ints" );
//    kernel->input( 1, &boardSize );
//    kernel->input( 1, &filterSize );
//    kernel->input( N * boardSize * boardSize, &(boards[0][0][0]) );
//    kernel->input( filterSize * filterSize, &(filter[0][0]) );
//    kernel->output( N * boardSize * boardSize, &(results[0][0][0]) );
//    kernel->run_1d( N * boardSize * boardSize, workgroupsize );
//    delete kernel;
//    timer.timeCheck("after convolve");

//    BoardPng::writeBoardsToPng( "testarraysquare-4.png", results, min(N, 100), boardSize );

//    BoardHelper::deleteBoard( &filter, filterSize );

    BoardsHelper::deleteBoards( &results, N, boardSize );
    BoardsHelper::deleteBoards( &boards, N, boardSize );
}

int main( int argc, char *argv[] ) {
//    clewInit();
//    MemoryChecker memoryChecker;
    run( argc, argv );
    return 0;
}


