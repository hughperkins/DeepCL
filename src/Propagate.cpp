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

Propagate::Propagate( OpenCLHelper *cl, LayerDimensions layerDimensions ) :
        cl( cl ),
        dim( layerDimensions ) {
}
STATIC Propagate *Propagate::instance(OpenCLHelper *cl, LayerDimensions dim ) {
    return new PropagateAuto( cl, dim );
//    return new PropagateByInputPlane( cl, dim );

//    if( dim.filterSize == dim.inputImageSize && dim.padZeros == false && dim.numFilters >= 64
//        && dim.filterSize >= 11 ) {
//        return new PropagateFc( cl, dim );
//    } else {
//    }
//    if( dim.filterSize == dim.inputImageSize && dim.padZeros == false && dim.numFilters >= 64
//        && dim.filterSize >= 11 ) {
//        return new PropagateFc( cl, dim );
//    } else if( square( dim.outputImageSize ) < 32 || square( dim.outputImageSize ) > cl->getMaxWorkgroupSize() ) {
//        return new Propagate1( cl, dim );
//    } else {
//        return new Propagate3( cl, dim );
//    }
}
STATIC Propagate *Propagate::instanceTest(OpenCLHelper *cl, LayerDimensions layerDimensions ) {
    return new Propagate1( cl, layerDimensions );
}
STATIC int Propagate::getNumImplementations() {
    return 7;
}
STATIC bool Propagate::plausiblyOptimal( int index, int batchSize, LayerDimensions dim ) {
    if( index == 0 ) { 
        return false;
    }
    if( index > 6 ) {
        return false;
    }
    return true;
}
STATIC Propagate *Propagate::instanceSpecific( int idx, OpenCLHelper *cl, LayerDimensions layerDimensions ) {
    if( idx == 0 ) {
        return new PropagateCpu( cl, layerDimensions );
    } else if( idx == -1 ) {
        return instance( cl, layerDimensions );
    } else if( idx == -2 ) {
        cout << "Propagate::instanceSpeicfic, choosing: PropagateAuto" << endl;
        return new PropagateAuto( cl, layerDimensions );
    } else if( idx == 1 ) {
        return new Propagate1( cl, layerDimensions );
    } else if( idx == 2 ) {
        return new Propagate2( cl, layerDimensions );
    } else if( idx == 3 ) {
        return new Propagate3( cl, layerDimensions );
    } else if( idx == 4 ) {
        return new Propagate4( cl, layerDimensions );
    } else if( idx == 5 ) {
        return new PropagateFc( cl, layerDimensions );
    } else if( idx == 6 ) {
        return new PropagateByInputPlane( cl, layerDimensions );
    } else if( idx == 99 ) {
        return new PropagateExperimental( cl, layerDimensions );
    } else {
        throw runtime_error( string("") + __FILE__ + ":" + toString( __LINE__ ) + " Propagate::instanceSpecific: no instance defined for index " + toString(idx) );
    }
}
STATIC Propagate *Propagate::instanceSpecific( std::string name, OpenCLHelper *cl, LayerDimensions layerDimensions ) {
    if( name == "cpu" ) {
        return new PropagateCpu( cl, layerDimensions );
    } else if( name == "prop1" ) {
        return new Propagate1( cl, layerDimensions );
    } else if( name == "prop3" ) {
        return new Propagate3( cl, layerDimensions );
    } else if( name == "prop4" ) {
        return new Propagate4( cl, layerDimensions );
    } else if( name == "fc" ) {
        return new PropagateFc( cl, layerDimensions );
    } else if( name == "byinplane" ) {
        return new PropagateByInputPlane( cl, layerDimensions );
    } else if( name == "exp" ) {
        return new PropagateExperimental( cl, layerDimensions );
    } else {
        throw runtime_error( string("") + __FILE__ + ":" + toString( __LINE__ ) + " Propagate::instanceSpecific: no instance defined for name " + name );
    }
}
// you own the returned output array, and are responsible for deleting it
VIRTUAL float * Propagate::forward( int batchSize, float *inputData, float *filters, float *biases ) {
    float *output = new float[batchSize * dim.outputCubeSize];
    forward( batchSize, inputData, filters, biases, output );
    return output;
}
// must allocate output yourself before the call
VIRTUAL void Propagate::forward( int batchSize, float *inputData, float *filters, float *biases, float *output ) {
    StatefulTimer::timeCheck("Propagate::forward begin");
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

//    int outputDataSize = batchSize * dim.outputCubeSize;
//    cout << " batchsize " << batchSize << " " << dim << endl;
//    int allocatedOutputSize = std::max(5000, outputDataSize );
//    int allocatedOutputSize = outputDataSize;
//    float *output = new float[allocatedOutputSize];
    CLWrapper *outputWrapper = cl->wrap( batchSize * dim.outputCubeSize, output );
    cl->finish();

    StatefulTimer::timeCheck("Propagate::forward after copied to device");
    forward( batchSize, dataWrapper, weightsWrapper, biasWeightsWrapper,
            outputWrapper );
    StatefulTimer::timeCheck("Propagate::forward after call forward");
    outputWrapper->copyToHost();
    StatefulTimer::timeCheck("Propagate::forward after copytohost");
//    for( int i = 0; i < 20; i++ ) {
//        cout << "output[" << i << "]=" << output[i] << endl;
//    }
    delete outputWrapper;

    delete dataWrapper;
    delete weightsWrapper;
    if( dim.biased ) {
        delete biasWeightsWrapper;
    }

//    return output;
}

