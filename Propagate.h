#pragma once

#include <iostream>
#include <string>

#include "OpenCLHelper.h"
#include "ActivationFunction.h"
#include "LayerDimensions.h"

using namespace std;

//inline float square( float value ) {
//    return value * value;
//}

#define STATIC static
#define VIRTUAL virtual

class Propagate {
public:
    OpenCLHelper *cl;
    LayerDimensions dim;
    ActivationFunction const*fn;
//    Propagate( OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction *fn ) :
//            dim( layerDimensions ),
//            cl( cl ),
//            fn( fn ) {
//    }
//    static Propagate *instance(OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction *fn );
    virtual void propagate( int batchSize, 
        CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
        CLWrapper *resultsWrapper ) = 0;
// { throw std::runtime_error("propagate by wrappers not implemented for this Propagate type" }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: Propagate
    // cppfile: Propagate.cpp

    STATIC Propagate *instance(OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn );
    STATIC Propagate *instanceTest(OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn );
    STATIC Propagate *instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn );
    Propagate( OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const*fn );
    VIRTUAL float * propagate( int batchSize, float *inputData, float *filters, float *biases );

    // [[[end]]]

//    virtual float * propagate( int batchSize, float *inputData, float *filters, float *biases ) {
//        int inputDataSize = batchSize * dim.inputPlanes * square( dim.inputBoardSize );
//        CLWrapper *dataWrapper = cl->wrap( inputDataSize, inputData );
//        dataWrapper->copyToDevice();

//        int weightsSize = dim.inputPlanes * dim.numFilters * square( dim.filterSize );
//        CLWrapper *weightsWrapper = cl->wrap( weightsSize, filters );
//        weightsWrapper->copyToDevice();

//        CLWrapper *biasWeightsWrapper = 0;
//        if( dim.biased ) {
//            int biasWeightsWrapperSize = dim.numFilters;
//            biasWeightsWrapper = cl->wrap( biasWeightsWrapperSize, biases );
//        }

//        int outputDataSize = batchSize * dim.numFilters * square( dim.outputBoardSize );
//        int allocatedResultsSize = std::min(5000, outputDataSize );
//        float *results = new float[allocatedResultsSize];
//        CLWrapper *resultsWrapper = cl->wrap( allocatedResultsSize, results );

//        propagate( batchSize, dataWrapper, weightsWrapper, biasWeightsWrapper,
//                resultsWrapper );
//        resultsWrapper->copyToHost();

//        for( int i = 0; i < 20; i++ ) {
//            cout << "results[" << i << "]=" << results[i] << endl;
//        }

//        delete dataWrapper;
//        delete weightsWrapper;
//        if( dim.biased ) {
//            delete biasWeightsWrapper;
//        }
//        delete resultsWrapper;

//        return results;
//    }
};



