// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

#ifdef gOutputImageSize // for previous tests that dont define it
// workgroup id organized like: [outplane]
// local id organized like: [outrow][outcol]
// each thread iterates over: [imageid][upstreamplane][filterrow][filtercol]
// number workgroups = 32
// one filter plane takes up 5 * 5 * 4 = 100 bytes
// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)
// all filter cubes = 3.2KB * 32 = 102KB (too big)
// output are organized like [imageid][filterid][row][col]
// assumes filter is small, so filtersize * filterSize * inputPlanes * 4 < about 3KB
//                            eg 5 * 5 * 32 * 4 = 3.2KB => ok :-)
//                           but 28 * 28 * 32 * 4 = 100KB => less good :-P
void kernel forward_2_by_outplane( const int batchSize,
      global const float *images, global const float *filters, 
        #ifdef BIASED
            global const float*biases, 
        #endif
    global float *output,
    local float *_upstreamImage, local float *_filterCube ) {
    const int globalId = get_global_id(0);

//    const int evenPadding = gFilterSize % 2 == 0 ? 1 : 0;

    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int outPlane = workgroupId;

    const int localId = get_local_id(0);
    const int outputRow = localId / gOutputImageSize;
    const int outputCol = localId % gOutputImageSize;

    #if gPadZeros == 1
        const int minu = max( -gHalfFilterSize, -outputRow );
        const int maxu = min( gHalfFilterSize, gOutputImageSize - 1 - outputRow ) - gEven;
        const int minv = max( -gHalfFilterSize, -outputCol );
        const int maxv = min( gHalfFilterSize, gOutputImageSize - 1 - outputCol ) - gEven;
    #else
        const int minu = -gHalfFilterSize;
        const int maxu = gHalfFilterSize - gEven;
        const int minv = -gHalfFilterSize;
        const int maxv = gHalfFilterSize - gEven;
    #endif

    const int numUpstreamsPerThread = ( gInputImageSizeSquared + workgroupSize - 1 ) / workgroupSize;

    const int filterCubeLength = gInputPlanes * gFilterSizeSquared;
    const int filterCubeGlobalOffset = outPlane * filterCubeLength;
    const int numPixelsPerThread = ( filterCubeLength + workgroupSize - 1 ) / workgroupSize;
    for( int i = 0; i < numPixelsPerThread; i++ ) {
        int thisOffset = localId + i * workgroupSize;
        if( thisOffset < filterCubeLength ) {
            _filterCube[thisOffset] = filters[filterCubeGlobalOffset + thisOffset];
        }
    }
    // dont need a barrier, since we'll just run behind the barrier from the upstream image download

    for( int n = 0; n < batchSize; n++ ) {
        float sum = 0;
        for( int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++ ) {
            int thisUpstreamImageOffset = ( n * gInputPlanes + upstreamPlane ) * gInputImageSizeSquared;
            barrier(CLK_LOCAL_MEM_FENCE);
            for( int i = 0; i < numUpstreamsPerThread; i++ ) {
                int thisOffset = workgroupSize * i + localId;
                if( thisOffset < gInputImageSizeSquared ) {
                    _upstreamImage[ thisOffset ] = images[ thisUpstreamImageOffset + thisOffset ];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            int filterImageOffset = upstreamPlane * gFilterSizeSquared;
            if( localId < gOutputImageSizeSquared ) {
                for( int u = minu; u <= maxu; u++ ) {
                    int inputRow = outputRow + u;
                    #if gPadZeros == 0
                         inputRow += gHalfFilterSize;
                    #endif
                    int inputimagerowoffset = inputRow * gInputImageSize;
                    int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                    for( int v = minv; v <= maxv; v++ ) {
                        int inputCol = outputCol + v;
                        #if gPadZeros == 0
                             inputCol += gHalfFilterSize;
                        #endif
                        sum += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                    }
                }
            }
        }
        #ifdef BIASED
            sum += biases[outPlane];
        #endif
        // output are organized like [imageid][filterid][row][col]
        int resultIndex = ( n * gNumFilters + outPlane ) * gOutputImageSizeSquared + localId;
        if( localId < gOutputImageSizeSquared ) {
            output[resultIndex ] = sum;
        }
    }
}
#endif

