
#include "OpenCLHelper.h"
#include "NeuralNet.h"
#include "Propagate.h"

#include "test/myasserts.h"
#include "test/WeightRandomizer.h"

#include <iostream>
#include <iomanip>

#include "gtest/gtest.h"

using namespace std;

#include "test/gtest_supp.h"

TEST( testpropagate, boardsize2_nopadzeros ) {
    int batchSize = 2;
    int numInPlanes = 1; int boardSize = 2;
    int numOutPlanes = 2; int filterWidth = 2;
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
    cout << "expected number of results: " << resultSize << endl;
    int outputBoardSize = 0;
    OpenCLHelper cl;
    Propagate *propagate = Propagate::instanceTest( &cl,
        LayerDimensions( numInPlanes, boardSize, numOutPlanes, filterWidth,
        padZeros == 1, false ), new LinearActivation() );
    float *results = propagate->propagate( batchSize, data, filter1, 0 );  
    for( int result = 0; result < resultSize; result++ ) {
        ASSERT_EQ( expectedResults[result], results[result] );
    }
    delete propagate;
    delete[] results;
}

TEST( testpropagate, boardsize2_padzeros ) {
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

    int outputBoardSize = 0;
    OpenCLHelper cl;
    Propagate *propagate = Propagate::instanceTest( &cl, LayerDimensions( numInPlanes, boardSize, numOutPlanes, filterWidth,
        padZeros == 1, false ), new LinearActivation() );
    float *results = propagate->propagate( batchSize, data, filter1, 0 );        

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
    delete propagate;
    delete[] results;
}

TEST( testpropagate, boardsize3 ) {
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

    int outputBoardSize = 0;
    OpenCLHelper cl;
    Propagate *propagate = Propagate::instanceTest( &cl, LayerDimensions( numInPlanes, boardSize, numOutPlanes, filterWidth,
        padZeros == 1, false ), new LinearActivation() );
    float *results = propagate->propagate( 
        batchSize, data, filter1, 0 );        

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
    delete propagate;
    delete[] results;
}

TEST( testpropagate, test2 ) {
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

    CLKernel *convolve = cl.buildKernel( "propagate.cl", "convolve_imagecubes_float2", "-D TANH" );
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

TEST( testpropagate, test3 ) {
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

    int outputBoardSize = 0;
    OpenCLHelper cl;
    Propagate *propagate = Propagate::instanceTest( &cl, LayerDimensions( numInPlanes, inBoardSize, numOutPlanes, filterSize,
        padZeros == 1, false ), new LinearActivation() );
    float *results = propagate->propagate( 
        batchSize, data, filter, 0 );        

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

TEST( testpropagate, boardsize19 ) {
    int batchSize = 128;
    int numInPlanes = 16;
    int numOutPlanes = 16;
    int inBoardSize = 19;
    int outBoardSize = 19;
    int filterSize = 5;
    int padZeros = 1;

    int inputSize = batchSize * numInPlanes * inBoardSize * inBoardSize;
    int resultsSize = batchSize * numOutPlanes * outBoardSize * outBoardSize;
    int weightsSize = numInPlanes * numOutPlanes * filterSize * filterSize;    
    int biasWeightsSize = numOutPlanes;
    float *inputs = new float[ inputSize ];
    float *filters = new float[weightsSize ];
    float *biasFilters = new float[biasWeightsSize];
    
    WeightRandomizer::randomize( inputs, inputSize );
    WeightRandomizer::randomize( filters, weightsSize );
    WeightRandomizer::randomize( biasFilters, biasWeightsSize );

    int outputBoardSize = 0;
    Timer timer;
    OpenCLHelper cl;
    Propagate *propagateImpl = Propagate::instanceTest( &cl, LayerDimensions( numInPlanes, inBoardSize, 
        numOutPlanes, filterSize, padZeros == 1, true ), new ReluActivation() );
    for( int i = 0; i < 10; i++ ) {
        float *results = propagateImpl->propagate( 
            batchSize, inputs, filters, biasFilters );
    }
    StatefulTimer::dump(true);
    timer.timeCheck("propagate time");
    delete propagateImpl;
}

TEST( testpropagate, DISABLED_dimensions_from_broken_mnist_layer_1 ) {
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

    int outputBoardSize = 0;
//    float *results = TestPropagateHelper::propagate( 
//        batchSize, 
//        numInPlanes, inBoardSize, 
//        numOutPlanes, filterSize,
//        &outputBoardSize,
//        padZeros == 1, true,
//        inputs, filters, biasFilters, new TanhActivation() );        
}

TEST( testpropagate, DISABLED_dimensions_from_broken_mnist_layer_2 ) {
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
    
    int outputBoardSize = 0;
//    float *results = TestPropagateHelper::propagate( 
//        batchSize, 
//        numInPlanes, inBoardSize, 
//        numOutPlanes, filterSize,
//        &outputBoardSize,
//        padZeros == 1, true,
//        inputs, filters, biasFilters, new TanhActivation() );        
}

TEST( testpropagate, mnist_finallayer ) {
    float *inputs = new float[ 10000 ];
    float *filters = new float[10000 ];
    float *biasFilters = new float[10000];

    memset( inputs, 0, sizeof(float) * 10000 );
    memset( filters, 0, sizeof(float) * 10000 );
    memset( biasFilters, 0, sizeof(float) * 10000 );

    WeightRandomizer::randomize( inputs, 10000, -0.1f, 0.1f );
    WeightRandomizer::randomize( filters, 10000, -0.1f, 0.1f );
    WeightRandomizer::randomize( biasFilters, 10000, -0.1f, 0.1f );

    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputBoardSize(28).setNumFilters( 10 ).setFilterSize( 28 )
        .setPadZeros( false ).setBiased( true );    

    OpenCLHelper cl;
    Propagate *p1 = Propagate::instance( &cl, dim, new TanhActivation() );
    for( int i = 0; i < (60000+batchSize - 1) / batchSize; i++ ) {
        float *results1 = p1->propagate( batchSize, inputs, filters, biasFilters );
        delete[] results1;
    }
    StatefulTimer::dump(true);

    delete p1;
}

TEST( testpropagate, mnist_intlayers_1024ex ) {
    float *inputs = new float[ 10000 ];
    float *filters = new float[10000 ];
    float *biasFilters = new float[10000];

    memset( inputs, 0, sizeof(float) * 10000 );
    memset( filters, 0, sizeof(float) * 10000 );
    memset( biasFilters, 0, sizeof(float) * 10000 );

    WeightRandomizer::randomize( inputs, 10000, -0.1f, 0.1f );
    WeightRandomizer::randomize( filters, 10000, -0.1f, 0.1f );
    WeightRandomizer::randomize( biasFilters, 10000, -0.1f, 0.1f );

    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputBoardSize(28).setNumFilters( 32 ).setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );    
    ActivationFunction *fn = new ReluActivation();

    OpenCLHelper cl;
    Propagate *p1 = Propagate::instance( &cl, dim, fn );
    for( int i = 0; i < (1024+batchSize - 1) / batchSize; i++ ) {
        float *results1 = p1->propagate( batchSize, inputs, filters, biasFilters );
        delete[] results1;
    }
    StatefulTimer::dump(true);

    delete p1;
}

TEST( testpropagate, mnist_firstconvlayer ) {
    float *inputs = new float[ 10000 ];
    float *filters = new float[10000 ];
    float *biasFilters = new float[10000];

    memset( inputs, 0, sizeof(float) * 10000 );
    memset( filters, 0, sizeof(float) * 10000 );
    memset( biasFilters, 0, sizeof(float) * 10000 );

    WeightRandomizer::randomize( inputs, 10000, -0.1f, 0.1f );
    WeightRandomizer::randomize( filters, 10000, -0.1f, 0.1f );
    WeightRandomizer::randomize( biasFilters, 10000, -0.1f, 0.1f );

    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 1 ).setInputBoardSize(28).setNumFilters( 32 ).setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );    
    ActivationFunction *fn = new ReluActivation();

    OpenCLHelper cl;
    Propagate *p1 = Propagate::instance( &cl, dim, fn );
    for( int i = 0; i < (60000+batchSize - 1) / batchSize; i++ ) {
        float *results1 = p1->propagate( batchSize, inputs, filters, biasFilters );
        delete[] results1;
    }
    StatefulTimer::dump(true);

    delete p1;
}

TEST( testpropagate, comparespecific ) {
    OpenCLHelper cl;
    float *inputs = new float[ 10000 ];
    float *filters = new float[10000 ];
    float *biasFilters = new float[10000];

    memset( inputs, 0, sizeof(float) * 10000 );
    memset( filters, 0, sizeof(float) * 10000 );
    memset( biasFilters, 0, sizeof(float) * 10000 );

    int batchSize = 1;
    LayerDimensions dim;
//    dim.setInputPlanes( 2 ).setInputBoardSize( 5 ).setNumFilters( 1 ).setFilterSize( 5 )
//        .setPadZeros( true ).setBiased( false );    
    dim.setInputPlanes( 2 ).setInputBoardSize(5).setNumFilters( 1 ).setFilterSize( 5 )
        .setPadZeros( true ).setBiased( false );    

    inputs[0] = 2;
//    inputs[4] = 7;
//    inputs[5] = 1;
    inputs[25] = 3;
    inputs[49] = 2;

//    filters[0] = 5;
//    filters[1] = 3;
//    filters[5] = 4;
    filters[24] = 11;
    filters[25] = 2;
      
    Propagate *p1 = Propagate::instanceSpecific( 1, &cl, dim, new LinearActivation() );
    float *results1 = p1->propagate( batchSize, inputs, filters, biasFilters );
    Propagate *p2 = Propagate::instanceSpecific( 3, &cl, dim, new LinearActivation() );
    float *results2 = p2->propagate( batchSize, inputs, filters, biasFilters );

    int resultsSize = batchSize * dim.outputCubeSize;
    cout << dim << endl;
    bool same = true;
    for( int i = 0; i < max( 50, resultsSize ); i++ ) {
        cout << "results[" << i << "]=" << results1[i] << " " << results2[i];
        if( i < resultsSize ) {
            if( results1[i] == results2[i] ) {
                cout << " SAME";
            } else {
                cout << " DIFF";
                same = false;
            }
        } else {
            cout << "     ";
        }
        cout << "  || " << results2[100+i] ;
        cout << "  || " << results2[200+i] ;
        cout << "  || " << results2[300+i] ;
        cout << "  || " << results2[400+i] ;
        cout << "  || " << results2[500+i] ;
        cout << "  || " << results2[600+i] ;
        cout << "  || " << results2[700+i] << endl;
    }
    EXPECT_EQ( true, same );
    delete[] results1;
    delete[] results2;
    delete p1;
    delete p2;
}

TEST( testpropagate, compare ) {
    OpenCLHelper cl;
    float *inputs = new float[ 10000 ];
    float *filters = new float[10000 ];
    float *biasFilters = new float[10000];
    WeightRandomizer::randomize( inputs, 10000 );
    WeightRandomizer::randomize( filters, 10000 );
    WeightRandomizer::randomize( biasFilters, 10000 );
    for( int batchSize = 1; batchSize <= 1; batchSize += 3 ) {
        for( int inputBoardSize = 5; inputBoardSize <= 5; inputBoardSize++ ) {
            for( int numInputPlanes = 2; numInputPlanes <= 2; numInputPlanes++ ) {
                for( int numFilters = 1; numFilters <= 1; numFilters++ ) {
                    for( int filterSize = 5; filterSize <= 5; filterSize++ ) {
                        for( int biased = 0; biased <= 1; biased++ ) {
                            for( int padZeros = 1; padZeros <= 1; padZeros++ ) {
                                int numInPlanes = numInputPlanes;
                                int numOutPlanes = numFilters;
                                LayerDimensions dim( numInputPlanes, inputBoardSize,
                                    numFilters, filterSize, padZeros == 1, biased == 1 );          
                                int inBoardSize = inputBoardSize;
                                int inputSize = batchSize * numInPlanes * inBoardSize * inBoardSize;
                                int resultsSize = batchSize * numOutPlanes * dim.outputBoardSize * dim.outputBoardSize;
                                int weightsSize = numInPlanes * numOutPlanes * filterSize * filterSize;    
                                cout << " OutputBoardSize " << dim.outputBoardSize << " resultsize " << resultsSize << endl;
                                int biasWeightsSize = numOutPlanes;
                                Propagate *p1 = Propagate::instanceSpecific( 1, &cl, dim, new LinearActivation() );
                                float *results1 = p1->propagate( batchSize, inputs, filters, biasFilters );
                                Propagate *p2 = Propagate::instanceSpecific( 3, &cl, dim, new LinearActivation() );
                                float *results2 = p2->propagate( batchSize, inputs, filters, biasFilters );
                                cout << " batchSize " + toString(batchSize ) +
                                    " inputBoardSize " + toString(inputBoardSize ) +
                                    " numInputPlanes=" + toString(numInputPlanes) +
                                    " numFilters=" + toString(numFilters) +
                                    " filterSize=" + toString(filterSize) +
                                    " biased=" + toString(biased) +
                                    " padZeros=" + toString(padZeros) << endl;
                                for( int i = 0; i < resultsSize; i++ ) {
                                    if( results1[i] != results2[i] ) {
                                        cout << "mismatch, i = " + toString(i) + 
                                            " batchSize " + toString(batchSize ) +
                                            " inputBoardSize " + toString(inputBoardSize ) +
                                            " numInputPlanes=" + toString(numInputPlanes) +
                                            " numFilters=" + toString(numFilters) +
                                            " filterSize=" + toString(filterSize) +
                                            " biased=" + toString(biased) +
                                            " padZeros=" + toString(padZeros) << endl;
                                        for( int i = 0; i < 2 * resultsSize; i++ ) {
                                            cout << "results[" << i << "]=" << results1[i] << " " << results2[i] << std::endl;
                                        }
                                        throw std::runtime_error("mismatch, i = " + toString(i) + 
                                            " batchSize " + toString(batchSize ) +
                                            " inputBoardSize " + toString(inputBoardSize ) +
                                            " numInputPlanes=" + toString(numInputPlanes) +
                                            " numFilters=" + toString(numFilters) +
                                            " filterSize=" + toString(filterSize) +
                                            " biased=" + toString(biased) +
                                            " padZeros=" + toString(padZeros) );
                                    }
                                }
                                delete[] results1;
                                delete[] results2;
                                delete p1;
                                delete p2;
                            }
                        }
                     }
                }
            }
        }
    }
}


