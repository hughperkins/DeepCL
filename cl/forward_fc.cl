// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#ifdef gOutImageSize // for previous tests that dont define it
// workgroupid [n][outputplane]
// localid: [filterrow][filtercol]
//  each thread iterates over: [inplane]
// this kernel assumes:
//   padzeros == 0 (mandatory)
//   filtersize == inputimagesize (mandatory)
//   outputImageSize == 1
//   lots of outplanes, hundreds, but less than max work groupsize, eg 350, 500, 361
//   lots of inplanes, eg 32
//   inputimagesize around 19, not too small
#if gFilterSize == gInputImageSize && gPadZeros == 0
void kernel forward_filter_matches_inimage( const int batchSize,
      global const float *images, global const float *filters,
    global float *output,
    local float *_upstreamImage, local float *_filterImage ) {
    const int globalId = get_global_id(0);

    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int n = workgroupId / gNumOutPlanes;
    const int outPlane = workgroupId % gNumOutPlanes;

    const int localId = get_local_id(0);
    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    float sum = 0;
    for( int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++ ) {
        int thisUpstreamImageOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamImageSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < numUpstreamsPerThread; i++ ) {
            int thisOffset = workgroupSize * i + localId;
            if( thisOffset < gUpstreamImageSizeSquared ) {
                _upstreamImage[ thisOffset ] = images[ thisUpstreamImageOffset + thisOffset ];
            }
        }
        const int filterGlobalOffset = ( outPlane * gUpstreamNumPlanes + upstreamPlane ) * gFilterSizeSquared;
        for( int i = 0; i < numFilterPixelsPerThread; i++ ) {
            int thisOffset = workgroupSize * i + localId;
            if( thisOffset < gFilterSizeSquared ) {
                _filterCube[thisOffset] = filters[filterGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if( localId < gNumFilters ) {
            for( int u = minu; u <= maxu; u++ ) {
                int inputRow = outputRow + u + ( gPadZeros ? 0 : gHalfFilterSize );
                int inputimagerowoffset = inputRow * gUpstreamImageSize;
                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                for( int v = minv; v <= maxv; v++ ) {
                    int inputCol = outputCol + v + ( gPadZeros ? 0 : gHalfFilterSize );
                    sum += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                }
            }
        }
    }
    // output are organized like [imageid][filterid][row][col]
    int resultIndex = ( n * gNumOutPlanes + outPlane ) * gOutImageSizeSquared + localId;
    if( localId < gNumFilters ) {
        output[resultIndex ] = sum;
    }
}
#endif
#endif


