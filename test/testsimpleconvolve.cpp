
#include "OpenCLHelper.h"
#include "NeuralNet.h"
#include "test/myasserts.h"

#include <iostream>
#include <iomanip>

#include "gtest/gtest.h"

using namespace std;

TEST( testsimpleconvolve, boardsize2_nopadzeros ) {
    int batchSize = 2;
    int numOutPlanes = 2;
    int numInPlanes = 1;
    int boardSize = 2;
    int filterWidth = 2;
    int padZeros = 0;

    float data[] = { 0, 0, 
                      0.5f, 0.5f,

                        13, 17,
                       -19, 2.3f,
};

    float filter1[] = { 0, 0,
                        -0.5f, 0.5f,

                        0.2f, 0.3f, 
                         0.7f, -1.1f,

 };
    int resultSize = 4;
    float expectedResults[] = {
        -0.5f * 0.5f + 0.5f * 0.5f,
        0.7f * 0.5f -1.1f * 0.5f,
        (-0.5f) * (-19) + 0.5f * 2.3f,
        0.2f*13 + 0.3f* 17 + 0.7f *(-19) -1.1f * 2.3f 
    };

    OpenCLHelper cl;
    float *results = new float[512];

    CLWrapper *dataWrapper = cl.wrap( batchSize * 9, data );
    CLWrapper *weightsWrapper = cl.wrap( numOutPlanes * 9, filter1 );
    CLWrapper *resultsWrapper = cl.wrap( 512, results );
    dataWrapper->copyToDevice();
    weightsWrapper->copyToDevice();

    CLKernel *kernel = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2", "-D LINEAR" );
    kernel->in(batchSize)->in( numInPlanes )->in( numOutPlanes )->in( boardSize )->in( filterWidth )
       ->in( padZeros );
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    kernel->output( resultsWrapper );
    int globalSize = batchSize * numOutPlanes * boardSize * boardSize;
    int workgroupsize = cl.getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    resultsWrapper->copyToHost();

    for( int result = 0; result < 4; result++ ) {
        cout << "results[" << result << "]=" << results[result] << endl;
    }

    for( int result = 0; result < resultSize; result++ ) {
        ASSERT_EQ( expectedResults[result], results[result] );
    }
}

TEST( testsimpleconvolve, boardsize2_padzeros ) {
    int batchSize = 2;
    int numOutPlanes = 2;
    int numInPlanes = 1;
    int boardSize = 2;
    int filterWidth = 2;
    int padZeros = 1;

    float data[] = { 0, 0, 
                      0.5f, 0.3f,

                        13, 17,
                       -19, 2.3f,
};

    float filter1[] = { 0, 0,
                        -0.5f, 0.4f,

                        0.2f, 0.3f, 
                         0.7f, -1.1f,

 };
    int resultSize = (boardSize + 1) * (boardSize + 1) * batchSize * numOutPlanes;
    float *expectedResults = new float[resultSize];
    for( int i = 0; i < resultSize; i++ ) {
        expectedResults[i] = -9999; // means havent provided an expectedresult.
    }

    expectedResults[0] = 0; expectedResults[1] = 0; expectedResults[2] = 0;

    expectedResults[3] = 0.5f*0.4f;
    expectedResults[4] = 0.5f*(-0.5f)+0.4f*(0.3f);
    expectedResults[5] = 0.3 * (-0.5f); 

    expectedResults[6] = 0; expectedResults[7] = 0; expectedResults[8] = 0;

    expectedResults[9] = 0; expectedResults[10] = 0; expectedResults[11] = 0;
    expectedResults[12] =(-1.1f)*0.5;
    expectedResults[13] = 0.7f * 0.5f + (-1.1f) * 0.3f;
    expectedResults[14] = 0.7f * 0.3f;

    // plane 2, filter 2 ...
    expectedResults[27] = (-1.1f*13);
    expectedResults[28] = 0.7f * 13 + (-1.1f)*17;
    expectedResults[29] = 0.7f*17;
    expectedResults[35] = 0.2f* 2.3f;

//    expectedResults[] = 0;
//    expectedResults[5] = 0;
//    expectedResults[6] = 0.3f * 0.5f;
//    expectedResults[7] = 0.2f * 0.5f;

//    expectedResults[8] = 13 * 0.5f;
//    expectedResults[9] = 17 * (-0.5f);
//    expectedResults[10] = (-19) * 0;
//    expectedResults[11] = 2.3f * 0;
// 
//    expectedResults[12] = 13 * (-1.1f);
//    expectedResults[13] = 17 * 0.7f;
//    expectedResults[14] = (-19) * 0.3f;
//    expectedResults[15] = 2.3f * 0.2f;

//        -0.5f * 0.5f + 0.5f * 0.5f,
//        0.7f * 0.5f -1.1f * 0.5f,
//        (-0.5f) * (-19) + 0.5f * 2.3f,
//        0.2f*13 + 0.3f* 17 + 0.7f *(-19) -1.1f * 2.3f 
//    };

    OpenCLHelper cl;
    float *results = new float[2048];

    CLWrapper *dataWrapper = cl.wrap( 8, data );
    CLWrapper *weightsWrapper = cl.wrap( 8, filter1 );
    CLWrapper *resultsWrapper = cl.wrap( 2048, results );
    dataWrapper->copyToDevice();
    weightsWrapper->copyToDevice();

    CLKernel *kernel = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2", "-D LINEAR" );
    kernel->in(batchSize)->in( numInPlanes )->in( numOutPlanes )->in( boardSize )->in( filterWidth )
       ->in( padZeros );
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    kernel->output( resultsWrapper );
    int globalSize = batchSize * numOutPlanes * boardSize * boardSize;
    int workgroupsize = cl.getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    resultsWrapper->copyToHost();

    for( int result = 1024; result < 1024 + 4; result++ ) {
        cout << "results[" << result << "]=" << results[result] << endl;
    }

//    ASSERT_EQ( -0.5f * 0.5f + 0.5f * 0.5f, results[0] );
//    ASSERT_EQ( 0.7f * 0.5f -1.1f * 0.5f, results[1] );
//    ASSERT_EQ( (-0.5f) * (-19) + 0.5f * 2.3f, results[2] );
//    ASSERT_EQ( 0.2f*13 + 0.3f* 17 + 0.7f *(-19) -1.1f * 2.3f , results[3] );

    for( int result = 0; result < resultSize; result++ ) {
        if( expectedResults[result] != -9999 ) {
            cout << " checking result[" << result << "]=" << results[result] << " expecting: " << expectedResults[result] << endl;
            ASSERT_FLOAT_EQ( expectedResults[result], results[result] );
        }
    }
}

TEST( testsimpleconvolve, boardsize3 ) {
    int batchSize = 5;
    int numOutPlanes = 2;
    int numInPlanes = 1;
    int boardSize = 3;
    int filterWidth = 3;
    int padZeros = 0;

    float data[] = { 0, 0, 0,
                       0, 0, 0,
                       0.5f, 0, 0.5f,

                        0, 0, 0,
                       0, 0, 0,
                       0.5f, 0, -0.5f ,

                        0, 0, 0,
                       0, 0, 0,
                       0.5f, 0, 0,

                        0, 0, 0,
                       0, 0, 0,
                       1, 10, 0,

                        0, 0, 0,
                       0, 0, 0,
                       0, 0, 1 
};

    float filter1[] = { 0, 0, 0,
                          0, 0, 0,
                         -0.5f, 0, 0.5f,

                        0, 0, 0,
                          0, 0, 0,
                         2.0f, 0.5, 0.5f,

 };

    OpenCLHelper cl;
    float *results = new float[512];

    CLWrapper *dataWrapper = cl.wrap( batchSize * 9, data );
    CLWrapper *weightsWrapper = cl.wrap( numOutPlanes * 9, filter1 );
    CLWrapper *resultsWrapper = cl.wrap( 512, results );
    dataWrapper->copyToDevice();
    weightsWrapper->copyToDevice();

    CLKernel *kernel = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2", "-D LINEAR" );
    kernel->in(batchSize)->in( numInPlanes )->in( numOutPlanes )->in( boardSize )->in( filterWidth )
       ->in( padZeros );
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    kernel->output( resultsWrapper );
    int globalSize = batchSize * numOutPlanes * boardSize * boardSize;
    int workgroupsize = cl.getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    resultsWrapper->copyToHost();

//    for( int i = 0; i < 20; i++ ) {
//        cout << "results[" << i << "]=" << results[i] << endl;
//    }
    assertEquals( 0, results[0] );
    assertEquals( 1.25f, results[1] );
    assertEquals( -0.5f, results[2] );
    assertEquals( 0.75f, results[3] );
    assertEquals( -0.25f, results[4] );
    assertEquals( 1, results[5] );
    assertEquals( -0.5f, results[6] );
    assertEquals( 7, results[7] );
    assertEquals( 0.5f, results[8] );
    assertEquals( 0.5f, results[9] );
        cout << "test1 ok" << endl;
}

TEST( testsimpleconvolve, test2 ) {
    int batchSize = 2;
    int numOutPlanes = 2;
    int numInPlanes = 1;
    int boardSize = 3;
    int filterWidth = 3;
    int padZeros = 0;

    float data[] = { 0, 0, 0,
                       -0.5f, 0.5f, 0,
                       0, 0, 0,

                        0, 0, 0,
                       0.5f, -0.5f, 0,
                       0, 0, 0

};

    float filter1[] = { 0, 0, 0,
                          0.300809, -0.11011, 0,
                         0, 0, 0,

                        0, 0, 0,
                          0.0570846, 0.347077, 0,
                         0,0,0

 };

    OpenCLHelper cl;
    float *results = new float[512];

    CLWrapper *dataWrapper = cl.wrap( batchSize * 9, data );
    CLWrapper *weightsWrapper = cl.wrap( numOutPlanes * 9, filter1 );
    CLWrapper *resultsWrapper = cl.wrap( 512, results );
    dataWrapper->copyToDevice();
    weightsWrapper->copyToDevice();

    CLKernel *convolve = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2", "-D TANH" );
//    CLKernel *tanh = cl.buildKernel( "ClConvolve.cl", "byelement_tanh" );

    for( int it = 0; it < 100; it ++ ) {
        convolve->in(batchSize)->in( numInPlanes )->in( numOutPlanes )->in( boardSize )->in( filterWidth )
           ->in( padZeros );
        convolve->input( dataWrapper );
        convolve->input( weightsWrapper);
        convolve->output( resultsWrapper );
        int globalSize = batchSize * numOutPlanes * boardSize * boardSize;
        int workgroupsize = cl.getMaxWorkgroupSize();
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//        cout << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
        convolve->run_1d( globalSize, workgroupsize );

        resultsWrapper->copyToHost();

//        for( int i = 0; i < 20; i++ ) {
//            cout << "results[" << i << "]=" << results[i] << endl;
//        }
        assertEquals( -0.202616f, results[0], 0.0001 );
        assertEquals( 0.143989f, results[1], 0.0001 );
        assertEquals( 0.202616f, results[2], 0.0001 );
        assertEquals( -0.143989f, results[3], 0.0001 );
    }
    cout << "test2 ok" << endl;
}

TEST( testsimpleconvolve, test3 ) {
    int batchSize = 4;
    int numInPlanes = 2;
    int numOutPlanes = 2;
    int inBoardSize = 1;
    int outBoardSize = 1;
    int filterSize = 1;
    int padZeros = 0;
    float data[] = {0.1,0.2,
                    0.3,0.4,
                    0.5,0.6,
                    0.7,0.8};
    float filter[] = {0.2,0.3,
                     0.5,0.7};
    float results[512];
    
    OpenCLHelper cl;
    CLWrapper *dataWrapper = cl.wrap( batchSize * numInPlanes * inBoardSize, data );
    CLWrapper *weightsWrapper = cl.wrap( numInPlanes * numOutPlanes * filterSize * filterSize, filter );
    CLWrapper *resultsWrapper = cl.wrap( 512, results );
    dataWrapper->copyToDevice();
    weightsWrapper->copyToDevice();

    CLKernel *convolve = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2", "-D LINEAR" );
    convolve->in(batchSize)->in( numInPlanes )->in( numOutPlanes )->in( inBoardSize )->in( filterSize )
       ->in( padZeros );
    convolve->input( dataWrapper );
    convolve->input( weightsWrapper);
    convolve->output( resultsWrapper );
    int globalSize = batchSize * numOutPlanes * outBoardSize * outBoardSize;
    int workgroupsize = cl.getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    convolve->run_1d( globalSize, workgroupsize );

    resultsWrapper->copyToHost();
//    for( int i = 0; i < 20; i++ ) {
//        cout << "results[" << fixed << setprecision(4   ) << i << "]=" << results[i] << endl;
//    }
//    cout << setprecision(4) << endl;

    float expectedResults[] = {0.2*0.1+0.3*0.2,
                               0.5*0.1+0.7*0.2,

                               0.2*0.3+0.3*0.4,
                               0.5*0.3+0.7*0.4,

                                0.2*0.5+0.3*0.6,
                               0.5*0.5+0.7*0.6,
 
                              0.2*0.7+0.3*0.8,
                               0.5*0.7+0.7*0.8
  };
   for( int i = 0; i < 8; i++ ) {
//      cout << " checking result " << i << endl;
//        cout << "results[" << i << "]=" << results[i] << endl;
      assertEquals( expectedResults[i], results[i], 0.0001);
   }
}

TEST( testsimpleconvolve, dimensions_from_broken_mnist_layer_1 ) {
    int batchSize = 128;
    int numInPlanes = 1;
    int numOutPlanes = 14;
    int inBoardSize = 28;
    int outBoardSize = 24;
    int filterSize = 5;
    int padZeros = 0;

    int inputSize = batchSize * numInPlanes * inBoardSize * inBoardSize;
    int resultsSize = batchSize * numOutPlanes * outBoardSize * outBoardSize;
    int weightsSize = numInPlanes * numOutPlanes * filterSize * filterSize;    
    int biasWeightsSize = numOutPlanes;
    float *inputs = new float[ inputSize ];
    float *filters = new float[weightsSize ];
    float *biasFilters = new float[biasWeightsSize];
    float *results = new float[resultsSize];;
    
    OpenCLHelper cl;
    CLWrapper *dataWrapper = cl.wrap( inputSize, inputs );
    CLWrapper *weightsWrapper = cl.wrap( weightsSize, filters );
    CLWrapper *biasWeightsWrapper = cl.wrap( biasWeightsSize, biasFilters );
    CLWrapper *resultsWrapper = cl.wrap( resultsSize, results );
    dataWrapper->copyToDevice();
    weightsWrapper->copyToDevice();
    biasWeightsWrapper->copyToDevice();

    CLKernel *convolve = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2", "-D TANH -D BIASED" );
    convolve->in(batchSize)->in( numInPlanes )->in( numOutPlanes )->in( inBoardSize )->in( filterSize )
       ->in( padZeros );
    convolve->input( dataWrapper );
    convolve->input( weightsWrapper);
    convolve->input( biasWeightsWrapper);
    convolve->output( resultsWrapper );
    int globalSize = resultsSize;
    int workgroupsize = cl.getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    convolve->run_1d( globalSize, workgroupsize );

    resultsWrapper->copyToHost();
}

TEST( testsimpleconvolve, dimensions_from_broken_mnist_layer_2 ) {
    int batchSize = 128;
    int numInPlanes = 14;
    int numOutPlanes = 10;
    int inBoardSize = 28;
    int outBoardSize = 24;
    int filterSize = 24;
    int padZeros = 0;

    int inputSize = batchSize * numInPlanes * inBoardSize * inBoardSize;
    int resultsSize = batchSize * numOutPlanes * outBoardSize * outBoardSize;
    int weightsSize = numInPlanes * numOutPlanes * filterSize * filterSize;    
    int biasWeightsSize = numOutPlanes;
    float *inputs = new float[ inputSize ];
    float *filters = new float[weightsSize ];
    float *biasFilters = new float[biasWeightsSize];
    float *results = new float[resultsSize];;
    
    OpenCLHelper cl;
    CLWrapper *dataWrapper = cl.wrap( inputSize, inputs );
    CLWrapper *weightsWrapper = cl.wrap( weightsSize, filters );
    CLWrapper *biasWeightsWrapper = cl.wrap( biasWeightsSize, biasFilters );
    CLWrapper *resultsWrapper = cl.wrap( resultsSize, results );
    dataWrapper->copyToDevice();
    weightsWrapper->copyToDevice();
    biasWeightsWrapper->copyToDevice();

    CLKernel *convolve = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2", "-D TANH -D BIASED" );
    convolve->in(batchSize)->in( numInPlanes )->in( numOutPlanes )->in( inBoardSize )->in( filterSize )
       ->in( padZeros );
    convolve->input( dataWrapper );
    convolve->input( weightsWrapper);
    convolve->input( biasWeightsWrapper);
    convolve->output( resultsWrapper );
    int globalSize = resultsSize;
    int workgroupsize = cl.getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    convolve->run_1d( globalSize, workgroupsize );

    resultsWrapper->copyToHost();
}

TEST( testsimpleconvolve, backprop_weights_2 ) {
    const int batchSize = 1;
    const int upstreamBoardSize = 1;
    const int boardSize = 1;
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

    float data[] = { 3.0f };
    float errors[] = { 7.0f };
    float *results = new float[resultsSize]; // ignored, for LINEAR
//    float results[] = { 11.0f };
    float *weightChanges = new float[max(1,20)];

    float expectedResults[] = { - 3 * 7 };

    OpenCLHelper cl;
    CLKernel *kernel = cl.buildKernel("../ClConvolve.cl", "backprop_floats_2", options );
    
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
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( upstreamBoardSizeSquared );

    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
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


TEST( testsimpleconvolve, backprop_weights_2_boardsize2 ) {
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

    float expectedResults[] = { -3 * 7, - 13 * 2, // -21, -26
                                 -17*4,-19*4 };   // -68, -76

    OpenCLHelper cl;
    CLKernel *kernel = cl.buildKernel("../ClConvolve.cl", "backprop_floats_2", options );
    
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
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( upstreamBoardSizeSquared );

    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
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

TEST( testsimpleconvolve, backprop_weights_2_boardsize3_filtersize3 ) {
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
    kernel
       ->in(learningMultiplier)
       ->in( batchSize )
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( upstreamBoardSizeSquared );

    int globalSize = batchSize * upstreamNumPlanes * numPlanes * upstreamBoardSizeSquared;
//        int workgroupsize = cl->getMaxWorkgroupSize();
    cout << " ideal globalsize: " << globalSize << endl;
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();    
    for( int i = 0; i < 20; i++ ) {
        cout << "weightchanges[" << i << "]=" << weightChanges[i] << endl;
    }
    for( int i = 0; i < weightsSize; i++ ) {
//        if( expectedResults[i] != weightChanges[i] ) {
//            cout << "mismatch for i " << i << endl;
//            EXPECT_EQ( expectedResults[i], weightChanges[i] );
//        }
    }

    delete kernel;
}


