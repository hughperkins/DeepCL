// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "Propagate3.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL Propagate3::~Propagate3() {
    delete kernel;
    delete repeatedAdd;
    delete activate;
}
VIRTUAL void Propagate3::propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
    CLWrapper *resultsWrapper ) {
    StatefulTimer::timeCheck("Propagate3::propagate begin");
    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();
    int maxglobalId = 0;

    kernel->in(batchSize);
    kernel->input( dataWrapper );
    kernel->input( weightsWrapper);
//    if( dim.biased ) kernel->input( biasWeightsWrapper );
    kernel->output( resultsWrapper );
//    cout << "square(dim.outputBoardSize) " << square( dim.outputBoardSize ) << endl;
    kernel->localFloats( square( dim.inputBoardSize ) );
    kernel->localFloats( square( dim.filterSize ) * dim.inputPlanes );

    int workgroupsize = std::max( 32, square( dim.outputBoardSize ) ); // no point in wasting threads....
    int numWorkgroups = dim.numFilters * batchSize;
    int globalSize = workgroupsize * numWorkgroups;
//    cout << "propagate3 numworkgroups " << numWorkgroups << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
    kernel->run_1d( globalSize, workgroupsize );
    cl->finish();
    StatefulTimer::timeCheck("Propagate3::propagate after kernel1");

//    {
//        resultsWrapper->copyToHost();
//        float const *results = reinterpret_cast< float const *>( resultsWrapper->getHostArray() );
//        for( int i = 0; i < min( 64, resultsWrapper->size() ); i++ ) {
//            cout << "results[" << i << "]=" << results[i] << endl;
//        }
//    }

    if( dim.biased ) {
        repeatedAdd->in( batchSize * dim.numFilters * dim.outputBoardSize * dim.outputBoardSize )
            ->in( dim.numFilters )
            ->in( dim.outputBoardSize * dim.outputBoardSize )
            ->inout( resultsWrapper )->in( biasWeightsWrapper );
        maxglobalId = batchSize * dim.numFilters * dim.outputBoardSize * dim.outputBoardSize;
        numWorkgroups = ( maxglobalId + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
        repeatedAdd->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
        cl->finish();
        StatefulTimer::timeCheck("Propagate3::propagate after repeatedAdd");
    }

    activate->in( batchSize * dim.numFilters * dim.outputBoardSize * dim.outputBoardSize )
        ->inout( resultsWrapper );
    maxglobalId = batchSize * dim.numFilters * dim.outputBoardSize * dim.outputBoardSize;
    numWorkgroups = ( maxglobalId + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
    activate->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
    cl->finish();
    StatefulTimer::timeCheck("Propagate3::propagate after activate");

    StatefulTimer::timeCheck("Propagate3::propagate after call propagate");
}
Propagate3::Propagate3( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn ) :
        Propagate( cl, dim, fn )
            {

    if( square( dim.outputBoardSize ) > cl->getMaxWorkgroupSize() ) {
        throw runtime_error("cannot use propagate3, since outputboardsize * outputboardsize > maxworkgroupsize");
    }

    std::string options = "-D " + fn->getDefineName();
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel3( "kernel", "cl/propagate3.cl", "propagate_3_by_n_outplane", 'options' )
    // stringify.write_kernel3( "repeatedAdd", "cl/per_element_add.cl", "repeated_add", 'options' )
    // stringify.write_kernel3( "activate", "cl/activate.cl", "activate", 'options' )
    // ]]]
    // generated using cog:
    const char * kernelSource =  R"DELIM(

    // Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
    //
    // This Source Code Form is subject to the terms of the Mozilla Public License,
    // v. 2.0. If a copy of the MPL was not distributed with this file, You can
    // obtain one at http://mozilla.org/MPL/2.0/.

    // concept: each workgroup handles convolving one input example with one filtercube
    // and writing out one single output plane
    //
    // workgroup id organized like: [imageid][outplane]
    // local id organized like: [outrow][outcol]
    // each thread iterates over: [upstreamplane][filterrow][filtercol]
    // number workgroups = 32
    // one filter plane takes up 5 * 5 * 4 = 100 bytes
    // one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)
    // all filter cubes = 3.2KB * 32 = 102KB (too big)
    // results are organized like [imageid][filterid][row][col]
    void kernel propagate_3_by_n_outplane( const int batchSize,
          global const float *images, global const float *filters,
        global float *results,
        local float *_upstreamBoard, local float *_filterCube ) {
        const int globalId = get_global_id(0);

        const int workgroupId = get_group_id(0);
        const int workgroupSize = get_local_size(0);
        const int n = workgroupId / gNumFilters;
        const int outPlane = workgroupId % gNumFilters;

        const int localId = get_local_id(0);
        const int outputRow = localId / gOutputBoardSize;
        const int outputCol = localId % gOutputBoardSize;

        const int minu = gPadZeros ? max( -gHalfFilterSize, -outputRow ) : -gHalfFilterSize;
        const int maxu = gPadZeros ? min( gHalfFilterSize - gEven, gOutputBoardSize - 1 - outputRow  - gEven) : gHalfFilterSize - gEven;
        const int minv = gPadZeros ? max( -gHalfFilterSize, -outputCol ) : - gHalfFilterSize;
        const int maxv = gPadZeros ? min( gHalfFilterSize - gEven, gOutputBoardSize - 1 - outputCol - gEven) : gHalfFilterSize - gEven;

        const int numUpstreamsPerThread = ( gInputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;

        const int filterCubeLength = gInputPlanes * gFilterSizeSquared;
        const int filterCubeGlobalOffset = outPlane * filterCubeLength;
        const int numPixelsPerThread = ( filterCubeLength + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numPixelsPerThread; i++ ) {
            int thisOffset = localId + i * workgroupSize;
            if( thisOffset < filterCubeLength ) {
                _filterCube[thisOffset] = filters[filterCubeGlobalOffset + thisOffset];
            }
        }
        // dont need a barrier, since we'll just run behind the barrier from the upstream board download

        float sum = 0;
        for( int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++ ) {
            int thisUpstreamBoardOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;
            barrier(CLK_LOCAL_MEM_FENCE);
            for( int i = 0; i < numUpstreamsPerThread; i++ ) {
                int thisOffset = workgroupSize * i + localId;
                if( thisOffset < gInputBoardSizeSquared ) {
                    _upstreamBoard[ thisOffset ] = images[ thisUpstreamBoardOffset + thisOffset ];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            int filterBoardOffset = upstreamPlane * gFilterSizeSquared;
            for( int u = minu; u <= maxu; u++ ) {
                int inputRow = outputRow + u;
                #if gPadZeros == 0
                    inputRow += gHalfFilterSize;
                #endif
                int inputboardrowoffset = inputRow * gInputBoardSize;
                int filterrowoffset = filterBoardOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                for( int v = minv; v <= maxv; v++ ) {
                    int inputCol = outputCol + v;
                    #if gPadZeros == 0
                        inputCol += gHalfFilterSize;
                    #endif
                    if( localId < gOutputBoardSizeSquared ) {
                        sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                    }
                }
            }
        }

        // results are organized like [imageid][filterid][row][col]
        int resultIndex = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared + localId;
        if( localId < gOutputBoardSizeSquared ) {
            results[resultIndex ] = sum;
        }
    }

    )DELIM";
    kernel = cl->buildKernelFromString( kernelSource, "propagate_3_by_n_outplane", options, "cl/propagate3.cl" );
    // generated using cog:
    const char * repeatedAddSource =  R"DELIM(

    // Copyright Hugh Perkins 2015 hughperkins at gmail
    //
    // This Source Code Form is subject to the terms of the Mozilla Public License,
    // v. 2.0. If a copy of the MPL was not distributed with this file, You can
    // obtain one at http://mozilla.org/MPL/2.0/.

    kernel void per_element_add( const int N, global float *target, global const float *source ) {
        const int globalId = get_global_id(0);
        if( globalId >= N ) {
            return;
        }
        target[globalId] += source[globalId];
    }

    // adds source to target
    // tiles source as necessary, according to tilingSize
    kernel void per_element_tiled_add( const int N, const int tilingSize, global float *target, global const float *source ) {
        const int globalId = get_global_id(0);
        if( globalId >= N ) {
            return;
        }
        target[globalId] += source[globalId % tilingSize];
    }

    kernel void repeated_add( const int N, const int sourceSize, const int repeatSize, global float *target, global const float *source ) {
        const int globalId = get_global_id(0);
        if( globalId >= N ) {
            return;
        }
        target[globalId] += source[ ( globalId / repeatSize ) % sourceSize ];
    }

    )DELIM";
    repeatedAdd = cl->buildKernelFromString( repeatedAddSource, "repeated_add", options, "cl/per_element_add.cl" );
    // generated using cog:
    const char * activateSource =  R"DELIM(

    // Copyright Hugh Perkins 2015 hughperkins at gmail
    //
    // This Source Code Form is subject to the terms of the Mozilla Public License,
    // v. 2.0. If a copy of the MPL was not distributed with this file, You can
    // obtain one at http://mozilla.org/MPL/2.0/.

    // expected defines:
    // one of: [ TANH | RELU | LINEAR | SIGMOID | SCALEDTANH ]

    #ifdef TANH
        #define ACTIVATION_FUNCTION(output) (tanh(output))
    #elif defined SCALEDTANH
        #define ACTIVATION_FUNCTION(output) ( 1.7159f * tanh( 0.66667f * output))
    #elif SIGMOID
        #define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))
    #elif defined RELU
        #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
    #elif defined LINEAR
        #define ACTIVATION_FUNCTION(output) (output)
    #endif

    #ifdef ACTIVATION_FUNCTION // protect against not defined
    kernel void activate( const int N, global float *inout ) {
        const int globalId = get_global_id(0);
        if( globalId >= N ) {
            return;
        }
        inout[globalId] = ACTIVATION_FUNCTION( inout[globalId] );
    }
    #endif

    )DELIM";
    activate = cl->buildKernelFromString( activateSource, "activate", options, "cl/activate.cl" );
    // [[[end]]]
}

