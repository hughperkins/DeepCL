#include <iostream>

#include "OpenCLHelper.h"
#include "NeuralNet.h"

using namespace std;

int main( int argc, char *argv[] ) {
    float *data = new float[8];
    data[0] = 0;
    data[1] = 0;
    data[2] = -0.5;
    data[3] = 0.5;
    data[4] = 0;
    data[5] = 0;
    data[6] = 0.5;
    data[7] = -0.5;

    float *filter1 = new float[4];
    filter1[0] = 0.5;
    filter1[1] = 0.5;
    filter1[2] = 0.5;
    filter1[3] = 0.5;

    float *bias1 = new float[1];
    bias1[0] = 0;

    OpenCLHelper cl;
    float *results = new float[1];

    CLWrapper *dataWrapper = cl.wrap( 8, data );
    CLWrapper *weightsWrapper = cl.wrap( 4, filter1 );
    CLWrapper *biasWrapper = cl.wrap( 1, bias1 );
    CLWrapper *resultsWrapper = cl.wrap( 1, results );
    dataWrapper->copyToDevice();
    weightsWrapper->copyToDevice();
    biasWrapper->copyToDevice();

    int batchSize = 2;
    int numPlanes = 1;
    int boardSize = 2;
    CLKernel *kernel = cl.buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2_withbias" );
    kernel->in( 1 )->in( 1 )->in( 2 )->in( 2 )->in( 0 );
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
    kernel->input( biasWrapper );
    kernel->output( resultsWrapper );
    int globalSize = batchSize * numPlanes * boardSize * boardSize;
    int workgroupsize = cl.getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    kernel->run_1d( globalSize, workgroupsize );
    resultsWrapper->copyToHost();

    for( int i = 0; i < 1; i++ ) {
        cout << "results[" << i << "]=" << results[i] << endl;
    }

    return 0;
}

