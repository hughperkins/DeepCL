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
    static float * propagate( int batchSize, 
                int inputPlanes, int inputBoardSize, 
                int numFilters, int filterSize, 
                int *p_outputBoardSize,
                bool padZeros, bool biased,
                float *inputData, float *filters, float *biases, ActivationFunction *fn ) {
        return propagate2( batchSize,
            inputPlanes, inputBoardSize, numFilters, filterSize, p_outputBoardSize, 
            padZeros, biased,
            inputData, filters, biases, fn );
    }
    static float * propagate1( int batchSize, 
                int inputPlanes, int inputBoardSize, 
                int numFilters, int filterSize, 
                int *p_outputBoardSize,
                bool padZeros, bool biased,
                float *inputData, float *filters, float *biases, ActivationFunction *fn ) {
        OpenCLHelper *cl = new OpenCLHelper();

        int inputDataSize = batchSize * inputPlanes * square( inputBoardSize );
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

        CLKernel *kernel = cl->buildKernel( "../ClConvolve.cl", "convolve_imagecubes_float2", "-D " + fn->getDefineName() );
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
    static float * propagate2( int batchSize, 
                int inputPlanes, int inputBoardSize, 
                int numFilters, int filterSize, 
                int *p_outputBoardSize,
                bool padZeros, bool biased,
                float *inputData, float *filters, float *biases, ActivationFunction *fn ) {
        OpenCLHelper *cl = new OpenCLHelper();

        int inputDataSize = batchSize * inputPlanes * square( inputBoardSize );
        CLWrapper *dataWrapper = cl->wrap( inputDataSize, inputData );
        dataWrapper->copyToDevice();

        int weightsSize = inputPlanes * numFilters * square( filterSize );
//        cout << "weightsize: " << weightsSize << endl;
        CLWrapper *weightsWrapper = cl->wrap( weightsSize, filters );
        weightsWrapper->copyToDevice();

        bool isEven = filterSize % 2 == 0;
        int outputBoardSize = padZeros ? ( isEven ? inputBoardSize + 1 : inputBoardSize ) : inputBoardSize - filterSize + 1;
        int outputDataSize = batchSize * numFilters * square( outputBoardSize );
        int allocatedResultsSize = std::max(100, outputDataSize );
        float *results = new float[allocatedResultsSize];
        CLWrapper *resultsWrapper = cl->wrap( allocatedResultsSize, results );

        std::string options = "-D " + fn->getDefineName();
        options += " -D gUpstreamBoardSize=" + toString(inputBoardSize);
        options += " -D gUpstreamBoardSizeSquared=" + toString(square(inputBoardSize));
        options += " -D gFilterSize=" + toString(filterSize);
        options += " -D gFilterSizeSquared=" + toString(square(filterSize));
        options += " -D gOutBoardSize=" + toString(outputBoardSize);
        options += " -D gOutBoardSizeSquared=" + toString(square(outputBoardSize));
        options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
        options += " -D gNumOutPlanes=" + toString(numFilters);
        options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
        options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
        options += " -D gUpstreamNumPlanes=" + toString(inputPlanes);
        CLKernel *kernel = cl->buildKernel( "../ClConvolve.cl", "convolve_imagecubes_float3", options );
        kernel->in(batchSize);
        kernel->input( dataWrapper );
        kernel->input( weightsWrapper);
        kernel->output( resultsWrapper );
//        cout << "square(outputBoardSize) " << square( outputBoardSize ) << endl;
        kernel->localFloats( square( inputBoardSize ) );
        kernel->localFloats( square( filterSize ) * numFilters );
        int workgroupsize = square( outputBoardSize );
        int numWorkgroups = numFilters;
        int globalSize = workgroupsize * numWorkgroups;
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

