// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// BIASED (or not)

#define getFilterImageOffset(filter, inputPlane) (( filter * gInputPlanes + inputPlane) * gFilterSizeSquared)
#define getResultImageOffset(n, filter) (( n * gNumFilters + filter) * gOutputSizeSquared)

// handle lower layer...
// gradOutput for upstream look like [n][inPlane][inRow][inCol]
// need to aggregate over: [outPlane][outRow][outCol] (?)
// need to backprop gradOutput along each possible weight
// each upstream feeds to:
//    - each of our filters (so numPlanes filters)
//    - each of our outpoint points (so imageSize * imageSize)
// gradOutput are provider per [n][inPlane][inRow][inCol]
// globalid is structured as: [n][upstreamPlane][upstreamRow][upstreamCol]
// there will be approx 128 * 32 * 28 * 28 = 3 million threads :-P
// grouped into 4608 workgroups
// maybe we want fewer than this?
// note: currently doesnt use bias as input.  thats probably an error?
void kernel calcGradInput( 
        const int upstreamNumPlanes, const int upstreamImageSize, const int filterSize, 
        const int outNumPlanes, const int outImageSize,
        const int padZeros,
        global const float *weights, global const float *gradOutput, global float *gradInput) {
    int globalId = get_global_id(0);
    const int halfFilterSize = filterSize >> 1;
    const int margin = padZeros ? halfFilterSize : 0;

    const int upstreamImageSizeSquared = upstreamImageSize * upstreamImageSize;
    const int upstreamImage2dId = globalId / upstreamImageSizeSquared;

    const int intraImageOffset = globalId % upstreamImageSizeSquared;
    const int upstreamRow = intraImageOffset / upstreamImageSize;
    const int upstreamCol = intraImageOffset % upstreamImageSize;

    const int upstreamPlane = upstreamImage2dId % upstreamNumPlanes;
    const int n = upstreamImage2dId / upstreamNumPlanes;

    const int minFilterRow = max(0, upstreamRow + margin - (outImageSize - 1));
    const int maxFilterRow = min(filterSize - 1, upstreamRow + margin);
    const int minFilterCol = max(0, upstreamCol + margin - (outImageSize -1));
    const int maxFilterCol = min(filterSize - 1, upstreamCol + margin);

    float sumWeightTimesOutError = 0;
    // aggregate over [outPlane][outRow][outCol]
    for (int outPlane = 0; outPlane < outNumPlanes; outPlane++) {
        for (int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {
            int outRow = upstreamRow + margin - filterRow;
            for (int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {
                int outCol = upstreamCol + margin - filterCol;
                int resultIndex = (( n * outNumPlanes 
                          + outPlane) * outImageSize
                          + outRow) * outImageSize
                          + outCol;
                float thisError = gradOutput[resultIndex];
                int thisWeightIndex = (( outPlane * upstreamNumPlanes
                                    + upstreamPlane) * filterSize
                                    + filterRow) * filterSize
                                    + filterCol;
                float thisWeight = weights[thisWeightIndex];
                float thisWeightTimesError = thisWeight * thisError;
                sumWeightTimesOutError += thisWeightTimesError;
            }
        }
    }
    gradInput[globalId] = sumWeightTimesOutError;
}

// as calcGradInput, but with local cache
// convolve weights with gradOutput to produce gradInput
// workgroupid: [n][inputPlane]
// localid: [upstreamrow][upstreamcol]
// per-thread aggregation: [outPlane][filterRow][filterCol]
// need to store locally:
// - _errorImage. size = outputSizeSquared
// - _filterImage. size = filtersizesquared
// note: currently doesnt use bias as input.  thats probably an error?
// inputs: gradOutput :convolve: filters => gradInput
//
// per workgroup:
// gradOutput: [outPlane][outRow][outCol] 32 * 19 * 19 * 4 = 46KB
// weights: [filterId][filterRow][filterCol] 32 * 5 * 5 * 4 = 3.2KB
#ifdef gOutputSize // for previous tests that dont define it
void kernel calcGradInputCached( 
        const int batchSize,
        global const float *gradOutputGlobal,
        global const float *filtersGlobal, 
        global float *gradInput,
        local float *_errorImage, 
        local float *_filterImage) {

    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int n = workgroupId / gInputPlanes;
    const int upstreamPlane = workgroupId % gInputPlanes;

    const int upstreamRow = localId / gInputSize;
    const int upstreamCol = localId % gInputSize;

    const int minFilterRow = max(0, upstreamRow + gMargin - (gOutputSize - 1));
    const int maxFilterRow = min(gFilterSize - 1, upstreamRow + gMargin);
    const int minFilterCol = max(0, upstreamCol + gMargin - (gOutputSize -1));
    const int maxFilterCol = min(gFilterSize - 1, upstreamCol + gMargin);

    const int filterPixelCopiesPerThread = (gFilterSizeSquared + workgroupSize - 1) / workgroupSize;
    const int errorPixelCopiesPerThread = (gOutputSizeSquared + workgroupSize - 1) / workgroupSize;
    const int pixelCopiesPerThread = max(filterPixelCopiesPerThread, errorPixelCopiesPerThread);

    float sumWeightTimesOutError = 0;
    for (int outPlane = 0; outPlane < gNumFilters; outPlane++) {
        const int filterImageGlobalOffset =(outPlane * gInputPlanes + upstreamPlane) * gFilterSizeSquared;
        const int errorImageGlobalOffset = (n * gNumFilters + outPlane) * gOutputSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < pixelCopiesPerThread; i++) {
            int thisOffset = workgroupSize * i + localId;
            if (thisOffset < gFilterSizeSquared) {
                _filterImage[ thisOffset ] = filtersGlobal[ filterImageGlobalOffset + thisOffset ];
            }
            if (thisOffset < gOutputSizeSquared) {
                _errorImage[ thisOffset ] = gradOutputGlobal[ errorImageGlobalOffset + thisOffset ];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
//        if (globalId == 0) {
//            for (int i = 0; i < gFilterSizeSquared; i++) {
//                gradInput[ (outPlane+1)*100 + i ] = _filterImage[i];
//            }
//        }
        for (int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {
            int outRow = upstreamRow + gMargin - filterRow;
            for (int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {
                int outCol = upstreamCol + gMargin - filterCol;
                int resultIndex = outRow * gOutputSize + outCol;
                float thisError = _errorImage[resultIndex];
                int thisWeightIndex = filterRow * gFilterSize + filterCol;
                float thisWeight = _filterImage[thisWeightIndex];
                float thisWeightTimesError = thisWeight * thisError;
                sumWeightTimesOutError += thisWeightTimesError;
            }
        }
    }
    const int upstreamImageGlobalOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;
    if (localId < gInputSizeSquared) {
        gradInput[upstreamImageGlobalOffset + localId] = sumWeightTimesOutError;
    }
}
#endif

// how about we make each workgroup handle one upstream plane, and iterate over examples?
// for now we assume that a workgroup is large enough to have one thread per location
// but we could always simply make each thread handle two pixels I suppose :-)
// so, workgroupId is [upstreamPlane]
// localId is [upstreamRow][upstreamCol]
// we iterate over [n]
#ifdef gOutputSize // for previous tests that dont define it
/*
void kernel calcGradInput2( 
        const int batchSize,
        global const float *weightsGlobal, global const float *gradOutputGlobal, 
        global float *gradInputGlobal,
        local float *_weightImage, local float *_errorImage) {
    const int globalId = get_global_id(0);
    const int workgroupId = get_group_id(0);
    const int localId = get_local_id(0);
    const int workgroupSize = get_local_size(0);

    const int upstreamPlane = workgroupId;
    const int upstreamRow = localId / gInputSize;
    const int upstreamCol = localId % gInputSize;

    const int 
    if (localId < filterSizeSquared) {
        _weightImage[localId] = weightsGlobal[localId];
    }

    for (int n = 0; n < batchSize; n++) {
        float sumWeightTimesOutError = 0;
        // aggregate over [outPlane][outRow][outCol]
        for (int outPlane = 0; outPlane < outNumPlanes; outPlane++) {
            for (int outRow = 0; outRow < outImageSize; outRow++) {
                // need to derive filterRow and filterCol, given outRow and outCol
                int filterRow = upstreamRow + margin - outRow;
                for (int outCol = 0; outCol < outImageSize; outCol++) {
                   // need to derive filterRow and filterCol, given outRow and outCol
                    int filterCol = upstreamCol + margin - outCol;
                    int resultIndex = (( n * outNumPlanes 
                              + outPlane) * outImageSize
                              + outRow) * outImageSize
                              + outCol;
                    float thisError = gradOutput[resultIndex];
                    int thisWeightIndex = (( outPlane * upstreamNumPlanes
                                        + upstreamPlane) * filterSize
                                        + filterRow) * filterSize
                                        + filterCol;
                    float thisWeight = weights[thisWeightIndex];
                    float thisWeightTimesError = thisWeight * thisError;
                    sumWeightTimesOutError += thisWeightTimesError;
                }
            }
        }
        gradInput[globalId] = sumWeightTimesOutError;
    }
}
*/
#endif

// so, we're just going to convolve the errorcubes with our filter cubes...
// like forward, but easier, since no activation function, and no biases
// errorcubes (*) filters => gradOutput
// for propagation we had:
//   images are organized like [imageId][plane][row][col]
//   filters are organized like [filterid][inplane][filterrow][filtercol]
//   output are organized like [imageid][filterid][row][col]
//   global id is organized like output, ie: [imageid][filterid][row][col]
//   - no local memory used currently
//   - each thread:
//     - loads a whole image
//     - loads a whole filter
//     - writes one output
// we will have the other way around:
//   errorcubes are organized like [imageid][outPlane][outRow][outCol]
//   filters are organized like [filterid][inplane][filterrow][filtercol]
//        (so we will swap filterid and inplane around when referencing filters, kindof)
//  globalid will be organized like upstreamoutput, ie [imageid][upstreamplane][upstreamrow][upstreamcol]
#ifdef gOutputSize // for previous tests that dont define it
void kernel convolve_errorcubes_float( 
       const int batchSize,
      global const float *errorcubes, global const float *filters, 
    global float *upstreamErrors) {
    int globalId = get_global_id(0);

    int upstreamImage2Id = globalId / gInputSizeSquared;
    int exampleId = upstreamImage2Id / gInputPlanes;
    int filterId = upstreamImage2Id % gInputPlanes;

    if (exampleId >= batchSize) {
        return;
    }
/*
    int errorCubeOffset = exampleId * gOutPlanes * gOutputSizeSquared;
    int filterCubeOffset = filterId * gNumInputPlanes * gFilterSizeSquared;

    int localid = globalId % upstreamImageSizeSquared;
    int upstreamRow = localid / gInputSize;
    int upstreamCol = localid % gInputSize;

    float sum = 0;
// ====in progress
    int minm = padZeros ? max(-halfFilterSize, -outputRow) : -halfFilterSize;
// ====to do
    int maxm = padZeros ? min(halfFilterSize, outputSize - 1 - outputRow) : halfFilterSize;
    int minn = padZeros ? max(-halfFilterSize, -outputCol) : - halfFilterSize;
    int maxn = padZeros ? min(halfFilterSize, outputSize - 1 - outputCol) : halfFilterSize;
    int inputPlane = 0;
    while(inputPlane < numInputPlanes) {
        int inputImageOffset = inputCubeOffset + inputPlane * inputSizeSquared;
        int filterImageOffset = filterCubeOffset + inputPlane * filterSizeSquared;
        int m = minm;
        while(m <= maxm) {
            int inputRow = outputRow + m + (padZeros ? 0 : halfFilterSize);
            int inputimagerowoffset = inputImageOffset + inputRow * inputSize;
            int filterrowoffset = filterImageOffset + (m+halfFilterSize) * filterSize + halfFilterSize;
            int n = minn;
            while(n <= maxn) {
                int inputCol = outputCol + n + (padZeros ? 0 : halfFilterSize);
                sum += images[ inputimagerowoffset + inputCol] * filters[ filterrowoffset + n ];
                n++;
            }
            m++;
        }
        inputPlane++;
    }
    output[globalId] = sum;*/
}
#endif

