#include <iostream>
#include <iomanip>

#include "OpenCLHelper.h"
#include "NeuralNet.h"
#include "BackpropWeights.h"

#include "test/myasserts.h"
#include "gtest/gtest.h"
#include "test/gtest_supp.h"

using namespace std;

TEST( testbackpropweights, backprop_weights_2 ) {
    LayerDimensions dim;
    dim.setInputBoardSize( 1 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );

    const int batchSize = 1;
    const float learningMultiplier = 1;

    float data[] = { 3.0f };
    float errors[] = { 7.0f };
    float *results = new float[batchSize * dim.outputCubeSize]; // ignored, for LINEAR
    float *weights = new float[max(dim.filtersSize,20)];
    float *biasWeights = new float[10];
    memset( weights, 0, sizeof( float ) * max( dim.filtersSize, 20 ) );
    memset( biasWeights, 0, sizeof(float) * 10 );

    float expectedResults[] = { - 3 * 7 };

    OpenCLHelper cl;
    BackpropWeights *backpropWeightsImpl = BackpropWeights::instanceForTest( &cl, dim, new LinearActivation() );
    backpropWeightsImpl->backpropWeights( batchSize, learningMultiplier, errors, results, data, weights, biasWeights );
    delete backpropWeightsImpl;
    
    for( int i = 0; i < 20; i++ ) {
        cout << "weights[" << i << "]=" << weights[i] << endl;
    }
    for( int i = 0; i < dim.filtersSize; i++ ) {
        if( expectedResults[i] != weights[i] ) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ( expectedResults[i], weights[i] );
        }
    }
    delete[] results;
    delete[] weights;
    delete[] biasWeights;
}


TEST( testbackpropweights, DISABLED_backprop_weights_2_upstreamboardsize2 ) {
    const int batchSize = 1;
    const int upstreamBoardSize = 2;
    const int boardSize = 2;
    const int filterSize = 1;
    const int upstreamNumPlanes = 1;
    const int numPlanes = 1;
    const int filterSizeSquared = filterSize * filterSize;
    const int boardSizeSquared = boardSize * boardSize;
    const int upstreamBoardSizeSquared = upstreamBoardSize * upstreamBoardSize;
    const int resultsSize = boardSizeSquared * numPlanes * batchSize;
    const int filtersSize = filterSizeSquared * numPlanes;
    const int upstreamResultsSize = upstreamBoardSizeSquared * upstreamNumPlanes * batchSize;
    const int weightsSize = upstreamNumPlanes * numPlanes * filterSizeSquared;
    const int padZeros = 0;
    const int biased = 0;

    const float learningMultiplier = 1;

    std::string options = " -D LINEAR";
    if( biased ) {
         options += " -D BIASED";
    }
    options += " -D gUpstreamBoardSize=" + toString(upstreamBoardSize);
    options += " -D gUpstreamBoardSizeSquared=" + toString(upstreamBoardSizeSquared);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gFilterSizeSquared=" + toString(filterSizeSquared);
    options += " -D gOutBoardSize=" + toString(boardSize);
    options += " -D gOutBoardSizeSquared=" + toString(boardSizeSquared);
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gNumOutPlanes=" + toString(numPlanes);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
    options += " -D gUpstreamNumPlanes=" + toString(upstreamNumPlanes);
    std::cout << "using kernel options: [" + options + "]" << std::endl;

    float data[] = { 3.0f, 13,
                    17, 19 };
//    float weights[] = { 5.0f };
    float errors[] = { 7.0f, 2,
                       4,4 };
    float *results = new float[resultsSize];
//    float results[] = { 11.0f, 2,
//                        5, 12 };
    float *weightChanges = new float[max(4,20)];

    float expectedResults[] = { -3 * 7 - 13 * 2 // -191
                                 -17*4 -19*4 };   // 

    OpenCLHelper cl;
    CLKernel *kernel = cl.buildKernel("../ClConvolve.cl", "backprop_floats_2", options );

    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    
    CLWrapper *imagesWrapper = cl.wrap( upstreamResultsSize, data );
    CLWrapper *resultsWrapper = cl.wrap( resultsSize, results );
    CLWrapper *errorsWrapper = cl.wrap( resultsSize, errors );
    CLWrapper *weightChangesWrapper = cl.wrap( max(weightsSize,20), weightChanges );
    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();
    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl.getNextPower2( workgroupsize ) )
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( workgroupsize );

    kernel->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();    
    for( int i = 0; i < 20; i++ ) {
        cout << "weightchanges[" << i << "]=" << weightChanges[i] << endl;
    }
    for( int i = 0; i < weightsSize; i++ ) {
        if( expectedResults[i] != weightChanges[i] ) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ( expectedResults[i], weightChanges[i] );
        }
    }

    delete kernel;
}

TEST( DISABLED_testsimpleconvolve, backprop_weights_2_upstreamboardsize3_filtersize3 ) {
    const int batchSize = 1;
    const int upstreamBoardSize = 3;
    const int boardSize = 1;
    const int filterSize = 3;
    const int upstreamNumPlanes = 1;
    const int numPlanes = 1;
    const int filterSizeSquared = filterSize * filterSize;
    const int boardSizeSquared = boardSize * boardSize;
    const int upstreamBoardSizeSquared = upstreamBoardSize * upstreamBoardSize;
    const int resultsSize = boardSizeSquared * numPlanes * batchSize;
    const int filtersSize = filterSizeSquared * numPlanes;
    const int upstreamResultsSize = upstreamBoardSizeSquared * upstreamNumPlanes * batchSize;
    const int weightsSize = upstreamNumPlanes * numPlanes * filterSizeSquared;
    const int padZeros = 0;
    const int biased = 0;

    const float learningMultiplier = 1;

    std::string options = " -D LINEAR";
    if( biased ) {
         options += " -D BIASED";
    }
    options += " -D gUpstreamBoardSize=" + toString(upstreamBoardSize);
    options += " -D gUpstreamBoardSizeSquared=" + toString(upstreamBoardSizeSquared);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gFilterSizeSquared=" + toString(filterSizeSquared);
    options += " -D gOutBoardSize=" + toString(boardSize);
    options += " -D gOutBoardSizeSquared=" + toString(boardSizeSquared);
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gNumOutPlanes=" + toString(numPlanes);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
    options += " -D gUpstreamNumPlanes=" + toString(upstreamNumPlanes);
    std::cout << "using kernel options: [" + options + "]" << std::endl;

    float data[] = { 3.0f, 13, 5,
                    17, 19, -3,
                    2, -4, 7 };
//    float weights[] = { 5.0f };
    float errors[] = { 7.0f };
    float *results = new float[resultsSize];
//    float results[] = { 11.0f, 2,
//                        5, 12 };
    float *weightChanges = new float[max(4,20)];

    float expectedResults[] = { -7 * 3, - 7 * 13, - 7 * 5, // -21 -91, -35
                                -7 * 17, - 7 * 19, 7 * 3,   // -119, 133, 21
                                - 7 * 2,  7 * 4, - 7 * 7 }; // -14, 28, -49

    OpenCLHelper cl;
    CLKernel *kernel = cl.buildKernel("../ClConvolve.cl", "backprop_floats_2", options );
    
    CLWrapper *imagesWrapper = cl.wrap( upstreamResultsSize, data );
    CLWrapper *resultsWrapper = cl.wrap( resultsSize, results );
    CLWrapper *errorsWrapper = cl.wrap( resultsSize, errors );
    CLWrapper *weightChangesWrapper = cl.wrap( max(weightsSize,20), weightChanges );
    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();

    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    cout << " ideal globalsize: " << globalSize << endl;
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl.getNextPower2( workgroupsize ) )
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( upstreamBoardSizeSquared );

    kernel->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();    
    for( int i = 0; i < 20; i++ ) {
        cout << "weightchanges[" << i << "]=" << weightChanges[i] << endl;
    }
    for( int i = 0; i < weightsSize; i++ ) {
        if( expectedResults[i] != weightChanges[i] ) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ( expectedResults[i], weightChanges[i] );
        }
    }

    delete kernel;
}

TEST( testbackpropweights, DISABLED_backprop_weights_2_upstreamboardsize4_filtersize3 ) {
    const int batchSize = 1;
    const int upstreamBoardSize = 4;
    const int boardSize = 2;
    const int filterSize = 3;
    const int upstreamNumPlanes = 1;
    const int numPlanes = 1;
    const int filterSizeSquared = filterSize * filterSize;
    const int boardSizeSquared = boardSize * boardSize;
    const int upstreamBoardSizeSquared = upstreamBoardSize * upstreamBoardSize;
    const int resultsSize = boardSizeSquared * numPlanes * batchSize;
    const int filtersSize = filterSizeSquared * numPlanes;
    const int upstreamResultsSize = upstreamBoardSizeSquared * upstreamNumPlanes * batchSize;
    const int weightsSize = upstreamNumPlanes * numPlanes * filterSizeSquared;
    const int padZeros = 0;
    const int biased = 0;

    const float learningMultiplier = 1;

    std::string options = " -D LINEAR";
    if( biased ) {
         options += " -D BIASED";
    }
    options += " -D gUpstreamBoardSize=" + toString(upstreamBoardSize);
    options += " -D gUpstreamBoardSizeSquared=" + toString(upstreamBoardSizeSquared);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gFilterSizeSquared=" + toString(filterSizeSquared);
    options += " -D gOutBoardSize=" + toString(boardSize);
    options += " -D gOutBoardSizeSquared=" + toString(boardSizeSquared);
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gNumOutPlanes=" + toString(numPlanes);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
    options += " -D gUpstreamNumPlanes=" + toString(upstreamNumPlanes);
    std::cout << "using kernel options: [" + options + "]" << std::endl;

    float data[] = { 3.0f, 13, 5, 8,
                    17, 19, -3, 2,
                    2, -4, 7, 0,
                    0, 6, 8, 9 };
//    float weights[] = { 5.0f };
    float errors[] = { 7.0f, 2,
                        0, -3 };
    float *results = new float[resultsSize];
//    float results[] = { 11.0f, 2,
//                        5, 12 };
    float *weightChanges = new float[max(4,20)];

    float expectedResults[] = { -3*7-13*2-0+19*3, -999, -999 , // 10
                                -999, -999, -999,
                                -999, -999, -49+27 };          //           -22

    OpenCLHelper cl;
    CLKernel *kernel = cl.buildKernel("../ClConvolve.cl", "backprop_floats_2", options );
    
    CLWrapper *imagesWrapper = cl.wrap( upstreamResultsSize, data );
    CLWrapper *resultsWrapper = cl.wrap( resultsSize, results );
    CLWrapper *errorsWrapper = cl.wrap( resultsSize, errors );
    CLWrapper *weightChangesWrapper = cl.wrap( max(weightsSize,20), weightChanges );
    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();

    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    cout << " ideal globalsize: " << globalSize << endl;
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl.getNextPower2( workgroupsize ) )
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( upstreamBoardSizeSquared );

    kernel->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();    
    for( int i = 0; i < 20; i++ ) {
        cout << "weightchanges[" << i << "]=" << weightChanges[i] << endl;
    }
    for( int i = 0; i < weightsSize; i++ ) {
        if( expectedResults[i] != -999 && expectedResults[i] != weightChanges[i] ) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ( expectedResults[i], weightChanges[i] );
        }
    }

    delete kernel;
}


TEST( testbackpropweights, DISABLED_backprop_weights_2_upstreamboardsize4_filtersize3_relu ) {
    const int batchSize = 1;
    const int upstreamBoardSize = 4;
    const int boardSize = 2;
    const int filterSize = 3;
    const int upstreamNumPlanes = 1;
    const int numPlanes = 1;
    const int filterSizeSquared = filterSize * filterSize;
    const int boardSizeSquared = boardSize * boardSize;
    const int upstreamBoardSizeSquared = upstreamBoardSize * upstreamBoardSize;
    const int resultsSize = boardSizeSquared * numPlanes * batchSize;
    const int filtersSize = filterSizeSquared * numPlanes;
    const int upstreamResultsSize = upstreamBoardSizeSquared * upstreamNumPlanes * batchSize;
    const int weightsSize = upstreamNumPlanes * numPlanes * filterSizeSquared;
    const int padZeros = 0;
    const int biased = 0;

    const float learningMultiplier = 1;

    std::string options = " -D RELU";
    if( biased ) {
         options += " -D BIASED";
    }
    options += " -D gUpstreamBoardSize=" + toString(upstreamBoardSize);
    options += " -D gUpstreamBoardSizeSquared=" + toString(upstreamBoardSizeSquared);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gFilterSizeSquared=" + toString(filterSizeSquared);
    options += " -D gOutBoardSize=" + toString(boardSize);
    options += " -D gOutBoardSizeSquared=" + toString(boardSizeSquared);
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gNumOutPlanes=" + toString(numPlanes);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
    options += " -D gUpstreamNumPlanes=" + toString(upstreamNumPlanes);
    std::cout << "using kernel options: [" + options + "]" << std::endl;

    float data[] = { 3.0f, 13, 5, 8,
                    17, 19, -3, 2,
                    2, -4, 7, 0,
                    0, 6, 8, 9 };
//    float weights[] = { 5.0f };
    float errors[] = { 7.0f, 2,
                        0, -3 };
//    float *results = new float[resultsSize];
    float results[] = { 11.0f, -2,
                        -5, 12 };
    float *weightChanges = new float[max(4,20)];

    float expectedResults[] = { -3*7-0*13*2-0+19*3, -999, -999 , // 36
                                -999, -999, -999,
                                -999, 4*7+3*8, -49+27 };          //           -22

    OpenCLHelper cl;
    CLKernel *kernel = cl.buildKernel("../ClConvolve.cl", "backprop_floats_2", options );
    
    CLWrapper *imagesWrapper = cl.wrap( upstreamResultsSize, data );
    CLWrapper *resultsWrapper = cl.wrap( resultsSize, results );
    CLWrapper *errorsWrapper = cl.wrap( resultsSize, errors );
    CLWrapper *weightChangesWrapper = cl.wrap( max(weightsSize,20), weightChanges );
    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();

    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    cout << " ideal globalsize: " << globalSize << endl;
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl.getNextPower2( workgroupsize ) )
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( upstreamBoardSizeSquared );

    kernel->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();    
    for( int i = 0; i < 20; i++ ) {
        cout << "weightchanges[" << i << "]=" << weightChanges[i] << endl;
    }
    for( int i = 0; i < weightsSize; i++ ) {
        if( expectedResults[i] != -999 && expectedResults[i] != weightChanges[i] ) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ( expectedResults[i], weightChanges[i] );
        }
    }

    delete kernel;
}

TEST( testbackpropweights, DISABLED_backprop_weights_2_upstreamboardsize5_filtersize3 ) {
    const int batchSize = 1;
    const int upstreamBoardSize = 5;
    const int boardSize = 3;
    const int filterSize = 3;
    const int upstreamNumPlanes = 1;
    const int numPlanes = 1;
    const int filterSizeSquared = filterSize * filterSize;
    const int boardSizeSquared = boardSize * boardSize;
    const int upstreamBoardSizeSquared = upstreamBoardSize * upstreamBoardSize;
    const int resultsSize = boardSizeSquared * numPlanes * batchSize;
    const int filtersSize = filterSizeSquared * numPlanes;
    const int upstreamResultsSize = upstreamBoardSizeSquared * upstreamNumPlanes * batchSize;
    const int weightsSize = upstreamNumPlanes * numPlanes * filterSizeSquared;
    const int padZeros = 0;
    const int biased = 0;

    const float learningMultiplier = 1;

    std::string options = " -D LINEAR";
    if( biased ) {
         options += " -D BIASED";
    }
    options += " -D gUpstreamBoardSize=" + toString(upstreamBoardSize);
    options += " -D gUpstreamBoardSizeSquared=" + toString(upstreamBoardSizeSquared);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gFilterSizeSquared=" + toString(filterSizeSquared);
    options += " -D gOutBoardSize=" + toString(boardSize);
    options += " -D gOutBoardSizeSquared=" + toString(boardSizeSquared);
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gNumOutPlanes=" + toString(numPlanes);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
    options += " -D gUpstreamNumPlanes=" + toString(upstreamNumPlanes);
    std::cout << "using kernel options: [" + options + "]" << std::endl;

    float data[] = { 3.0f, 13,  5, 8, 3,
                    17,    19, -3, 2, 1,
                    2,     -4,  7, 0, -2,
                    0,     6,   8, 9, 4,
                     1,   3,    5, 3, 8 };
//    float weights[] = { 5.0f };
    float errors[] = { 7.0f, 2,-1,
                        0, -3,1,
                        2,-1,0 };
    float *results = new float[resultsSize];
//    float results[] = { 11.0f, 2,
//                        5, 12 };
    float *weightChanges = new float[max(4,20)];

    float expectedResults[] = { -(3*7+13*2-1*5+0*17-3*19-1*3+2*2+1*4+0*7), -999, -999 , // 10
                                -999, -(19*7-3*2-2*1+  0-3*7+0*1   +2*6-1*8+0), -999,
                                -999, -999, -(7*7+0+2*1   +0-3*9+1*4   +5*2-1*3+0) };          //           -22

    OpenCLHelper cl;
    CLKernel *kernel = cl.buildKernel("../ClConvolve.cl", "backprop_floats_2", options );
    
    CLWrapper *imagesWrapper = cl.wrap( upstreamResultsSize, data );
    CLWrapper *resultsWrapper = cl.wrap( resultsSize, results );
    CLWrapper *errorsWrapper = cl.wrap( resultsSize, errors );
    CLWrapper *weightChangesWrapper = cl.wrap( max(weightsSize,20), weightChanges );
    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();

    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    cout << " ideal globalsize: " << globalSize << endl;
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl.getNextPower2( workgroupsize ) )
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( upstreamBoardSizeSquared );

    kernel->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();    
    for( int i = 0; i < 20; i++ ) {
        cout << "weightchanges[" << i << "]=" << weightChanges[i] << endl;
    }
    for( int i = 0; i < weightsSize; i++ ) {
        if( expectedResults[i] != -999 && expectedResults[i] != weightChanges[i] ) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ( expectedResults[i], weightChanges[i] );
        }
    }

    delete kernel;
}

TEST( testbackpropweights, DISABLED_backprop_weights_2_upstreamboardsize3_filtersize1 ) {
    const int batchSize = 1;
    const int upstreamBoardSize = 3;
    const int boardSize = 3;
    const int filterSize = 1;
    const int upstreamNumPlanes = 1;
    const int numPlanes = 1;
    const int filterSizeSquared = filterSize * filterSize;
    const int boardSizeSquared = boardSize * boardSize;
    const int upstreamBoardSizeSquared = upstreamBoardSize * upstreamBoardSize;
    const int resultsSize = boardSizeSquared * numPlanes * batchSize;
    const int filtersSize = filterSizeSquared * numPlanes;
    const int upstreamResultsSize = upstreamBoardSizeSquared * upstreamNumPlanes * batchSize;
    const int weightsSize = upstreamNumPlanes * numPlanes * filterSizeSquared;
    const int padZeros = 0;
    const int biased = 0;

    const float learningMultiplier = 1;

    cout << "upstreamBoardSizeSquare " << upstreamBoardSizeSquared << " upstreamresultssize " << upstreamResultsSize << " resultsSize " << resultsSize <<
        " weightsSize " << weightsSize << endl;    


    std::string options = " -D LINEAR";
    if( biased ) {
         options += " -D BIASED";
    }
    options += " -D gUpstreamBoardSize=" + toString(upstreamBoardSize);
    options += " -D gUpstreamBoardSizeSquared=" + toString(upstreamBoardSizeSquared);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gFilterSizeSquared=" + toString(filterSizeSquared);
    options += " -D gOutBoardSize=" + toString(boardSize);
    options += " -D gOutBoardSizeSquared=" + toString(boardSizeSquared);
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gNumOutPlanes=" + toString(numPlanes);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
    options += " -D gUpstreamNumPlanes=" + toString(upstreamNumPlanes);
    std::cout << "using kernel options: [" + options + "]" << std::endl;

    float *data = new float[ upstreamBoardSizeSquared ];
    memset( data, 0, sizeof(float) * upstreamBoardSizeSquared );

//    data[3 * upstreamBoardSize + 14] = 2;
    data[0] = 2;
    data[1 * upstreamBoardSize + 1] = 7;
    data[2 * upstreamBoardSize + 2] = 5;
//    data[8 * upstreamBoardSize + 15] = -2;

    float *errors = new float[ boardSizeSquared ];
    memset( errors, 0, sizeof(float) * boardSizeSquared );

//    errors[3 * upstreamBoardSize + 14] = 1;
    errors[0] = 5;
    errors[1 * boardSize + 1] = 11;
    errors[2 * boardSize + 2] = 3;
//    errors[8 * upstreamBoardSize + 15] = 7;

    float *results = new float[resultsSize];
    float *weightChanges = new float[max(4,512)];

    float expectedResults[] = { -(2 * 5 +  5 * 3 + 7 * 11 ) };          //           

    OpenCLHelper cl;
    CLKernel *kernel = cl.buildKernel("../ClConvolve.cl", "backprop_floats_2", options );

    CLWrapper *imagesWrapper = cl.wrap( upstreamResultsSize, data );
    CLWrapper *resultsWrapper = cl.wrap( resultsSize, results );
    CLWrapper *errorsWrapper = cl.wrap( resultsSize, errors );
    CLWrapper *weightChangesWrapper = cl.wrap( max(weightsSize,20), weightChanges );

    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();

    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    cout << " ideal globalsize: " << globalSize << endl;
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl.getNextPower2( workgroupsize ) )

       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )

        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( upstreamBoardSizeSquared );

    kernel->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();    
    for( int i = 0; i < 10; i++ ) {
        cout << "weightchanges[" << i << "]=" << weightChanges[i] << endl;
    }
    for( int i = 0; i < weightsSize; i++ ) {
        if( expectedResults[i] != -999 && expectedResults[i] != weightChanges[i] ) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ( expectedResults[i], weightChanges[i] );
        }
    }

    delete kernel;
}

TEST( testbackpropweights, DISABLED_backprop_weights_2_upstreamboardsize16_filtersize1 ) {
    const int batchSize = 1;
    const int upstreamBoardSize = 16;
    const int boardSize = 16;
    const int filterSize = 1;
    const int upstreamNumPlanes = 1;
    const int numPlanes = 1;
    const int filterSizeSquared = filterSize * filterSize;
    const int boardSizeSquared = boardSize * boardSize;
    const int upstreamBoardSizeSquared = upstreamBoardSize * upstreamBoardSize;
    const int resultsSize = boardSizeSquared * numPlanes * batchSize;
    const int filtersSize = filterSizeSquared * numPlanes;
    const int upstreamResultsSize = upstreamBoardSizeSquared * upstreamNumPlanes * batchSize;
    const int weightsSize = upstreamNumPlanes * numPlanes * filterSizeSquared;
    const int padZeros = 0;
    const int biased = 0;

    const float learningMultiplier = 1;

    std::string options = " -D LINEAR";
    if( biased ) {
         options += " -D BIASED";
    }
    options += " -D gUpstreamBoardSize=" + toString(upstreamBoardSize);
    options += " -D gUpstreamBoardSizeSquared=" + toString(upstreamBoardSizeSquared);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gFilterSizeSquared=" + toString(filterSizeSquared);
    options += " -D gOutBoardSize=" + toString(boardSize);
    options += " -D gOutBoardSizeSquared=" + toString(boardSizeSquared);
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gNumOutPlanes=" + toString(numPlanes);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
    options += " -D gUpstreamNumPlanes=" + toString(upstreamNumPlanes);
    std::cout << "using kernel options: [" + options + "]" << std::endl;

    float *data = new float[ upstreamBoardSizeSquared ];
    memset( data, 0, sizeof(float) * upstreamBoardSizeSquared );

//    data[3 * upstreamBoardSize + 14] = 2;
    data[0] = 2;
    data[15 * upstreamBoardSize + 15] = 5;
//    data[8 * upstreamBoardSize + 15] = -2;

    float *errors = new float[ boardSizeSquared ];
    memset( errors, 0, sizeof(float) * boardSizeSquared );

//    errors[3 * upstreamBoardSize + 14] = 1;
    errors[0] = 4;
    errors[15 * boardSize + 15] = 3;
//    errors[8 * upstreamBoardSize + 15] = 7;

    float *results = new float[resultsSize];
    float *weightChanges = new float[max(4,20)];

    float expectedResults[] = { -(2 * 4 +  3 * 5 ) };          //           

    OpenCLHelper cl;
    CLKernel *kernel = cl.buildKernel("../ClConvolve.cl", "backprop_floats_2", options );
    
    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    cout << " ideal globalsize: " << globalSize << endl;
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

    CLWrapper *imagesWrapper = cl.wrap( upstreamResultsSize, data );
    CLWrapper *resultsWrapper = cl.wrap( resultsSize, results );
    CLWrapper *errorsWrapper = cl.wrap( resultsSize, errors );
    CLWrapper *weightChangesWrapper = cl.wrap( max(weightsSize,20), weightChanges );
    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();
    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl.getNextPower2( workgroupsize ) )
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( upstreamBoardSizeSquared );

    kernel->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();    
    for( int i = 0; i < 20; i++ ) {
        cout << "weightchanges[" << i << "]=" << weightChanges[i] << endl;
    }
    for( int i = 0; i < weightsSize; i++ ) {
        if( expectedResults[i] != -999 && expectedResults[i] != weightChanges[i] ) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ( expectedResults[i], weightChanges[i] );
        }
    }

    delete kernel;
}

TEST( testbackpropweights, DISABLED_backprop_weights_2_upstreamboardsize17_filtersize1 ) {
    const int batchSize = 1;
    const int upstreamBoardSize = 17;
    const int boardSize = 17;
    const int filterSize = 1;
    const int upstreamNumPlanes = 1;
    const int numPlanes = 1;
    const int filterSizeSquared = filterSize * filterSize;
    const int boardSizeSquared = boardSize * boardSize;
    const int upstreamBoardSizeSquared = upstreamBoardSize * upstreamBoardSize;
    const int resultsSize = boardSizeSquared * numPlanes * batchSize;
    const int filtersSize = filterSizeSquared * numPlanes;
    const int upstreamResultsSize = upstreamBoardSizeSquared * upstreamNumPlanes * batchSize;
    const int weightsSize = upstreamNumPlanes * numPlanes * filterSizeSquared;
    const int padZeros = 0;
    const int biased = 0;

    const float learningMultiplier = 1;

    std::string options = " -D LINEAR";
    if( biased ) {
         options += " -D BIASED";
    }
    options += " -D gUpstreamBoardSize=" + toString(upstreamBoardSize);
    options += " -D gUpstreamBoardSizeSquared=" + toString(upstreamBoardSizeSquared);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gFilterSizeSquared=" + toString(filterSizeSquared);
    options += " -D gOutBoardSize=" + toString(boardSize);
    options += " -D gOutBoardSizeSquared=" + toString(boardSizeSquared);
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gNumOutPlanes=" + toString(numPlanes);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
    options += " -D gUpstreamNumPlanes=" + toString(upstreamNumPlanes);
    std::cout << "using kernel options: [" + options + "]" << std::endl;

    float *data = new float[ upstreamBoardSizeSquared ];
    memset( data, 0, sizeof(float) * upstreamBoardSizeSquared );

    data[0] = 2;
    data[1] = 3.2f;
    data[2] = 1.234f;
    data[16 * upstreamBoardSize + 16] = 5;

    float *errors = new float[ boardSizeSquared ];
    memset( errors, 0, sizeof(float) * boardSizeSquared );

    errors[0] = 4;
    errors[1] = -2.5f;
    errors[2] = 4.125f;
    errors[16 * boardSize + 16] = 3;

    float *results = new float[resultsSize];
    float *weightChanges = new float[max(4,20)];

    float expectedResults[] = { -( 4*2 - 3.2f * 2.5f + 1.234f * 4.125f + 3*5 ) };          // 

    OpenCLHelper cl;
    CLKernel *kernel = cl.buildKernel("../ClConvolve.cl", "backprop_floats_2", options );

    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    cout << " ideal globalsize: " << globalSize << endl;
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    
    CLWrapper *imagesWrapper = cl.wrap( upstreamResultsSize, data );
    CLWrapper *resultsWrapper = cl.wrap( resultsSize, results );
    CLWrapper *errorsWrapper = cl.wrap( resultsSize, errors );
    CLWrapper *weightChangesWrapper = cl.wrap( max(weightsSize,20), weightChanges );
    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();
    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl.getNextPower2( workgroupsize ) )
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( upstreamBoardSizeSquared );

    kernel->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();    
    for( int i = 0; i < 20; i++ ) {
        cout << "weightchanges[" << i << "]=" << weightChanges[i] << endl;
    }
    for( int i = 0; i < weightsSize; i++ ) {
        if( expectedResults[i] != -999 && expectedResults[i] != weightChanges[i] ) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ( expectedResults[i], weightChanges[i] );
        }
    }

    delete kernel;
}

TEST( testbackpropweights, DISABLED_backprop_weights_2_upstreamboardsize17_filtersize1_moredata ) {
    const int batchSize = 1;
    const int upstreamBoardSize = 17;
    const int boardSize = 17;
    const int filterSize = 1;
    const int upstreamNumPlanes = 1;
    const int numPlanes = 1;
    const int filterSizeSquared = filterSize * filterSize;
    const int boardSizeSquared = boardSize * boardSize;
    const int upstreamBoardSizeSquared = upstreamBoardSize * upstreamBoardSize;
    const int resultsSize = boardSizeSquared * numPlanes * batchSize;
    const int filtersSize = filterSizeSquared * numPlanes;
    const int upstreamResultsSize = upstreamBoardSizeSquared * upstreamNumPlanes * batchSize;
    const int weightsSize = upstreamNumPlanes * numPlanes * filterSizeSquared;
    const int padZeros = 0;
    const int biased = 0;

    const float learningMultiplier = 1;

    std::string options = " -D LINEAR";
    if( biased ) {
         options += " -D BIASED";
    }
    options += " -D gUpstreamBoardSize=" + toString(upstreamBoardSize);
    options += " -D gUpstreamBoardSizeSquared=" + toString(upstreamBoardSizeSquared);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gFilterSizeSquared=" + toString(filterSizeSquared);
    options += " -D gOutBoardSize=" + toString(boardSize);
    options += " -D gOutBoardSizeSquared=" + toString(boardSizeSquared);
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gNumOutPlanes=" + toString(numPlanes);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
    options += " -D gUpstreamNumPlanes=" + toString(upstreamNumPlanes);
    std::cout << "using kernel options: [" + options + "]" << std::endl;

    float *data = new float[ upstreamBoardSizeSquared ];
    memset( data, 0, sizeof(float) * upstreamBoardSizeSquared );
    for( int i = 0; i < upstreamBoardSizeSquared; i++ ) {
        data[i] = ( ( 1 + i ) % 20 ) / 5.3f;
    }

    float *errors = new float[ boardSizeSquared ];
    memset( errors, 0, sizeof(float) * boardSizeSquared );
    for( int i = 0; i < boardSizeSquared; i++ ) {
        errors[i] = ( ( 2 + i ) % 17 ) / 4.2f;
    }

    float *results = new float[resultsSize];
    float *weightChanges = new float[max(4,20)];

    float expectedResults[1];
    expectedResults[0] = 0;
    for ( int i = 0; i < upstreamBoardSizeSquared; i++ ) {
        expectedResults[0] += - data[i] * errors[i];
    }
    cout << "expectedresult: " << expectedResults[0] << endl;

    OpenCLHelper cl;
    CLKernel *kernel = cl.buildKernel("../ClConvolve.cl", "backprop_floats_2", options );
    
    CLWrapper *imagesWrapper = cl.wrap( upstreamResultsSize, data );
    CLWrapper *resultsWrapper = cl.wrap( resultsSize, results );
    CLWrapper *errorsWrapper = cl.wrap( resultsSize, errors );
    CLWrapper *weightChangesWrapper = cl.wrap( max(weightsSize,20), weightChanges );
    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();

    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    cout << " ideal globalsize: " << globalSize << endl;
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
//    int getNextPower2
    int workgroupsizepower2 = cl.getNextPower2( workgroupsize );
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << " workgroupsizepower2 "
        << workgroupsizepower2 << endl;

    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( workgroupsizepower2 )
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( upstreamBoardSizeSquared );

    kernel->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();    
    for( int i = 0; i < 20; i++ ) {
        cout << "weightchanges[" << i << "]=" << weightChanges[i] << endl;
    }
    for( int i = 0; i < weightsSize; i++ ) {
        if( expectedResults[i] != -999 ) {
//            cout << "mismatch for i " << i << endl;
            ASSERT_FLOAT_NEAR( expectedResults[i], weightChanges[i] );
        }
    }

    delete kernel;
}

