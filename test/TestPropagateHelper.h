#pragma once

#include <iostream>
#include <string>

#include "OpenCLHelper.h"

using namespace std;

//inline float square( float value ) {
//    return value * value;
//}

inline int square( int value ) {
    return value * value;
}

class TestPropagateHelper {
public:
//    static void propagate( 
//        int batchSize,
//        int inputBoardSize, float *inputData,
//        int filterSize, float *filterData,
//        int outputBoardSize, float *outputData,
//        bool padZeros
//    ) {
////        OpenCLHelper cl;
////        propagate1( batchSize, inputBoardSize, inputData, filterSize, filterData, outputBoardSize, outputData, padZeros );
//    }
    static float * propagate1( int batchSize, 
                int inputPlanes, int inputBoardSize, 
                int numFilters, int filterSize, 
                int *p_outputBoardSize,
                bool padZeros, bool biased,
                float *inputData, float *filters, float *biases, ActivationFunction *fn ) {
        OpenCLHelper *cl = new OpenCLHelper();

        int inputDataSize = batchSize * square( inputBoardSize );
        CLWrapper *dataWrapper = cl->wrap( inputDataSize, inputData );
        dataWrapper->copyToDevice();

        int weightsSize = inputPlanes * numFilters * square( filterSize );
        CLWrapper *weightsWrapper = cl->wrap( weightsSize, filters );
        weightsWrapper->copyToDevice();

        bool isEven = filterSize % 2 == 0;
        int outputBoardSize = padZeros ? ( isEven ? inputBoardSize + 1 : inputBoardSize ) : inputBoardSize - filterSize + 1;
        int outputDataSize = batchSize * numFilters * square( outputBoardSize );
        int allocatedResultsSize = std::min(5000, outputDataSize );
        float *results = new float[allocatedResultsSize];
        CLWrapper *resultsWrapper = cl->wrap( allocatedResultsSize, results );

        CLKernel *kernel = cl->buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2", "-D " + fn->getDefineName() );
        kernel->in(batchSize)->in( inputPlanes )->in( numFilters )->in( inputBoardSize )->in( filterSize )
           ->in( padZeros ? 1 : 0 );
        kernel->input( dataWrapper );
        kernel->input( weightsWrapper);
        kernel->output( resultsWrapper );

        int globalSize = batchSize * numFilters * square( outputBoardSize );
        int workgroupsize = std::min( globalSize, cl->getMaxWorkgroupSize() );
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
        cout << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;

        kernel->run_1d( globalSize, workgroupsize );
        cl->finish();
        resultsWrapper->copyToHost();

        for( int i = 0; i < 20; i++ ) {
            cout << "results[" << i << "]=" << results[i] << endl;
        }

//        memcpy( outputData, results, sizeof(float) * outputDataSize );
//        delete[]resmults;
        delete kernel;
        delete cl;
        return results;
    }
};

