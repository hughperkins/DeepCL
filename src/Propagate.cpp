// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <algorithm>

#include "Propagate.h"
#include "stringhelper.h"
#include "PropagateCpu.h"
#include "Propagate1.h"
#include "Propagate2.h"
#include "Propagate3.h"
#include "Propagate3_unfactorized.h"
#include "Propagate4.h"
#include "PropagateFc.h"
#include "PropagateByInputPlane.h"
#include "PropagateExperimental.h"
#include "PropagateAuto.h"
#include "StatefulTimer.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

STATIC Propagate *Propagate::instance(OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const *fn ) {
    return new PropagateAuto( cl, dim, fn );
//    return new PropagateByInputPlane( cl, dim, fn );

//    if( dim.filterSize == dim.inputBoardSize && dim.padZeros == false && dim.numFilters >= 64
//        && dim.filterSize >= 11 ) {
//        return new PropagateFc( cl, dim, fn );
//    } else {
//    }
//    if( dim.filterSize == dim.inputBoardSize && dim.padZeros == false && dim.numFilters >= 64
//        && dim.filterSize >= 11 ) {
//        return new PropagateFc( cl, dim, fn );
//    } else if( square( dim.outputBoardSize ) < 32 || square( dim.outputBoardSize ) > cl->getMaxWorkgroupSize() ) {
//        return new Propagate1( cl, dim, fn );
//    } else {
//        return new Propagate3( cl, dim, fn );
//    }
}
STATIC Propagate *Propagate::instanceTest(OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn ) {
    return new Propagate1( cl, layerDimensions, fn );
}
STATIC int Propagate::getNumImplementations() {
    return 7;
}
STATIC bool Propagate::plausiblyOptimal( int index, int batchSize, LayerDimensions dim, ActivationFunction const*fn ) {
    if( index == 0 ) { 
        return false;
    }
    if( index > 7 ) {
        return false;
    }
    return true;
}
STATIC Propagate *Propagate::instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn ) {
    if( idx == 0 ) {
        return new PropagateCpu( cl, layerDimensions, fn );
    } else if( idx == -1 ) {
        return instance( cl, layerDimensions, fn );
    } else if( idx == -2 ) {
        cout << "Propagate::instanceSpeicfic, choosing: PropagateAuto" << endl;
        return new PropagateAuto( cl, layerDimensions, fn );
    } else if( idx == 1 ) {
        return new Propagate1( cl, layerDimensions, fn );
    } else if( idx == 2 ) {
        return new Propagate2( cl, layerDimensions, fn );
    } else if( idx == 3 ) {
        return new Propagate3( cl, layerDimensions, fn );
    } else if( idx == 4 ) {
        return new Propagate4( cl, layerDimensions, fn );
    } else if( idx == 5 ) {
        return new PropagateFc( cl, layerDimensions, fn );
    } else if( idx == 6 ) {
        return new PropagateByInputPlane( cl, layerDimensions, fn );
    } else if( idx == 7 ) {
        return new Propagate3_unfactorized( cl, layerDimensions, fn );
    } else if( idx == 99 ) {
        return new PropagateExperimental( cl, layerDimensions, fn );
    } else {
        throw runtime_error( string("") + __FILE__ + ":" + toString( __LINE__ ) + " Propagate::instanceSpecific: no instance defined for index " + toString(idx) );
    }
}
STATIC Propagate *Propagate::instanceSpecific( std::string name, OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const *fn ) {
    if( name == "cpu" ) {
        return new PropagateCpu( cl, layerDimensions, fn );
    } else if( name == "prop1" ) {
        return new Propagate1( cl, layerDimensions, fn );
    } else if( name == "prop3" ) {
        return new Propagate3( cl, layerDimensions, fn );
    } else if( name == "prop4" ) {
        return new Propagate4( cl, layerDimensions, fn );
    } else if( name == "fc" ) {
        return new PropagateFc( cl, layerDimensions, fn );
    } else if( name == "byinplane" ) {
        return new PropagateByInputPlane( cl, layerDimensions, fn );
    } else if( name == "exp" ) {
        return new PropagateExperimental( cl, layerDimensions, fn );
    } else {
        throw runtime_error( string("") + __FILE__ + ":" + toString( __LINE__ ) + " Propagate::instanceSpecific: no instance defined for name " + name );
    }
}
Propagate::Propagate( OpenCLHelper *cl, LayerDimensions layerDimensions, ActivationFunction const*fn ) :
        dim( layerDimensions ),
        cl( cl ),
        fn( fn ) {
}
VIRTUAL float * Propagate::propagate( int batchSize, float *inputData, float *filters, float *biases ) {
    StatefulTimer::timeCheck("Propagate::propagate begin");
    int inputDataSize = batchSize * dim.inputCubeSize;
    CLWrapper *dataWrapper = cl->wrap( inputDataSize, inputData );
    dataWrapper->copyToDevice();

    int weightsSize = dim.filtersSize;
    CLWrapper *weightsWrapper = cl->wrap( weightsSize, filters );
    weightsWrapper->copyToDevice();

    CLWrapper *biasWeightsWrapper = 0;
    if( dim.biased ) {
        int biasWeightsWrapperSize = dim.numFilters;
        biasWeightsWrapper = cl->wrap( biasWeightsWrapperSize, biases );
        biasWeightsWrapper->copyToDevice();
    }

    int outputDataSize = batchSize * dim.outputCubeSize;
//    cout << " batchsize " << batchSize << " " << dim << endl;
    int allocatedResultsSize = std::max(5000, outputDataSize );
    float *results = new float[allocatedResultsSize];
    CLWrapper *resultsWrapper = cl->wrap( allocatedResultsSize, results );
    cl->finish();

    StatefulTimer::timeCheck("Propagate::propagate after copied to device");
    propagate( batchSize, dataWrapper, weightsWrapper, biasWeightsWrapper,
            resultsWrapper );
    StatefulTimer::timeCheck("Propagate::propagate after call propagate");
    resultsWrapper->copyToHost();
    StatefulTimer::timeCheck("Propagate::propagate after copytohost");
//    for( int i = 0; i < 20; i++ ) {
//        cout << "results[" << i << "]=" << results[i] << endl;
//    }
    delete resultsWrapper;

    delete dataWrapper;
    delete weightsWrapper;
    if( dim.biased ) {
        delete biasWeightsWrapper;
    }

    return results;
}

