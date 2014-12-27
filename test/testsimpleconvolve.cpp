#include <iostream>

#include "OpenCLHelper.h"
#include "NeuralNet.h"
#include "test/myasserts.h"

using namespace std;

void test1() {
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

    CLKernel *kernel = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2" );
    kernel->in( numInPlanes )->in( numOutPlanes )->in( boardSize )->in( filterWidth )
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

    for( int i = 0; i < 20; i++ ) {
        cout << "results[" << i << "]=" << results[i] << endl;
    }
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
}

void test2() {
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

    CLKernel *convolve = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2" );
    CLKernel *tanh = cl.buildKernel( "ClConvolve.cl", "byelement_tanh" );

    for( int it = 0; it < 100; it ++ ) {
        convolve->in( numInPlanes )->in( numOutPlanes )->in( boardSize )->in( filterWidth )
           ->in( padZeros );
        convolve->input( dataWrapper );
        convolve->input( weightsWrapper);
        convolve->output( resultsWrapper );
        int globalSize = batchSize * numOutPlanes * boardSize * boardSize;
        int workgroupsize = cl.getMaxWorkgroupSize();
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//        cout << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
        convolve->run_1d( globalSize, workgroupsize );

        tanh->inout( resultsWrapper );
        tanh->run_1d( globalSize, workgroupsize );

        resultsWrapper->copyToHost();

//        for( int i = 0; i < 20; i++ ) {
//            cout << "results[" << i << "]=" << results[i] << endl;
//        }
        assertEquals( -0.202616f, results[0] );
        assertEquals( 0.143989f, results[1] );
        assertEquals( 0.202616f, results[2] );
        assertEquals( -0.143989f, results[3] );
    }
}

int main( int argc, char *argv[] ) {
//    test1();
    test2();
    return 0;
}

