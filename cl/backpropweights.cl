// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR ]
// BIASED (or not)

#ifdef TANH
    #define ACTIVATION_DERIV(output) (1 - output * output)
#elif defined SCALEDTANH
    #define ACTIVATION_DERIV(output) ( 0.66667f * ( 1.7159f - 1 / 1.7159f * output * output) )
#elif defined RELU
    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : 0)
#elif defined LINEAR
    #define ACTIVATION_DERIV(output) (1)
#elif defined SIGMOID
    #define ACTIVATION_DERIV(output) (output * ( 1 - output) )
#endif

// images are organized like [imageId][plane][row][col]    128*32*19*19=1,500,000
// filters are organized like [filterid][inplane][filterrow][filtercol] 32*32*5*5=25600 = 100k bytes, or 3.2KB per filter
// results are organized like [imageid][filterid][row][col]   128*32*19*19=1,500,000 = 6MB, or 46KB per image,
//                                                            
//                  if w updates are per image,then 25600*128 = 3.3 million
// eg 32 * 32 * 5 * 5 = 25600 ...
// then we are aggregating over [outRow][outCol][n]
//      eg 19 * 19 * 128 = 46208
// derivtype: 0=relu 1=tanh
// outimages(eg 128x32x28x28), errors (eg 128x28x28), upstreamimages (eg 128x32x28x28) => weightchanges (eg 32x32x28x28)
// if break for per-example, per-filter:
// outimage(eg 28x28), error (28x28), upstreamimage(32x28x28) => weightchanges(32x5x5)
//             784 3k         784 3k                 25088 100k                800 3k
// if break for per-filter:
// outimage(eg 128x28x28), error (128x28x28), upstreamimage(128x32x28x28) => weightchanges(32x32x5x5)
//                350k           350k                 12.8MB                   100k
// if break for per-example:
// outimage(eg 28x28), error (28x28), upstreamimage(32x28x28) => weightchanges(32x5x5)
//                3k             3k                 100k                       3k
//    note that weightchanges will need to be summed over 128 input images
//
// globalid is for: [outPlane][upstreamPlane][filterRow][filterCol]
// per-thread looping over [n][outRow][outCol]
#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
void kernel backprop_floats( const float learningRateMultiplier,
        const int batchSize, const int upstreamNumPlanes, const int numPlanes, 
         const int upstreamImageSize, const int filterSize, const int outImageSize, const int padZeros, 
         global const float *errors, global const float *results, global const float *images, 
        global float *weights
        #ifdef BIASED
            , global float *biasWeights
        #endif
 ) {
    int globalId = get_global_id(0);
    if( globalId >= numPlanes * upstreamNumPlanes * filterSize * filterSize ) {
        return;
    }

    int filterSizeSquared = filterSize * filterSize;

    int IntraFilterOffset = globalId % filterSizeSquared;
    int filterRow = IntraFilterOffset / filterSize;
    int filterCol = IntraFilterOffset % filterSize;

    int filter2Id = globalId / filterSizeSquared;
    int outPlane = filter2Id / upstreamNumPlanes;
    int upstreamPlane = filter2Id % upstreamNumPlanes;

    const int halfFilterSize = filterSize >> 1;
    const int margin = padZeros ? halfFilterSize : 0;
    float thiswchange = 0;
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    for( int n = 0; n < batchSize; n++ ) {
        for( int outRow = 0; outRow < outImageSize; outRow++ ) {
            int upstreamRow = outRow - margin + filterRow;
            for( int outCol = 0; outCol < outImageSize; outCol++ ) {
                int upstreamCol = outCol - margin + filterCol;
                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < upstreamImageSize
                    && upstreamCol < upstreamImageSize;
                if( proceed ) {
                    int resultIndex = ( ( n * numPlanes 
                              + outPlane ) * outImageSize
                              + outRow ) * outImageSize
                              + outCol;
                    float error = errors[resultIndex];
                    float actualOutput = results[resultIndex];
                    float activationDerivative = ACTIVATION_DERIV( actualOutput);
                    int upstreamDataIndex = ( ( n * upstreamNumPlanes 
                                     + upstreamPlane ) * upstreamImageSize
                                     + upstreamRow ) * upstreamImageSize
                                     + upstreamCol;
                    float upstreamResult = images[upstreamDataIndex];
                    float thisimagethiswchange = upstreamResult * activationDerivative *
                        error;
                    thiswchange += thisimagethiswchange;
    #ifdef BIASED
                    thisbiaschange += activationDerivative;
    #endif
                }
            }
        }
    }
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    weights[ globalId ] += - learningRateMultiplier * thiswchange;
#ifdef BIASED
    bool writeBias = upstreamPlane == 0 && IntraFilterOffset == 0;
    if( writeBias ) {
        biasWeights[outPlane] += - learningRateMultiplier * thisbiaschange;
    }
#endif
}
#endif

#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
#ifdef gOutImageSize // for previous tests that dont define it
/*void kernel backprop_floats_2( 
    const float learningRateMultiplier, const int batchSize, const int workgroupsizenextpower2,
     global const float *upstreamImagesGlobal, global const float *resultsGlobal, global const float *errorsGlobal,
     global float *weightChangesGlobal,
    local float *_upstreamImage, local float *_resultImage, local float *_errorImage, 
    local float *_weightChanges, local float *_weightReduceArea ) {

        // required (minimum...) sizes for local arrays:
        // upstreamimage: upstreamImageSizeSquared
        // resultimage: outImageSizeSquared
        // errorImage: outImageSizeSquaread
        // weightChanges: filterSizeSquared
        // weightReduceArea: upstreamImageSizeSquared, or workflowSize, to be decided :-)
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupSize = get_local_size(0);

    const int upstreamImage2dId = globalId / gUpstreamImageSizeSquared;
    const int upstreamPlane = upstreamImage2dId % gUpstreamNumPlanes;
    const int outPlane2dId = upstreamImage2dId / gUpstreamNumPlanes;
    const int n = outPlane2dId / gNumOutPlanes;
    const int outPlane = outPlane2dId % gNumOutPlanes;

    const int outRow = localId / gOutImageSize;
    const int outCol = localId % gOutImageSize;

    // each localid corresponds to one [upstreamRow][upstreamCol] combination
    // we assume that:
    // filterSize <= upstreamImageSize (reasonable... :-) )
    // outImageSize <= upstreamImageSize (true... unless we have a filter with even size, and padZeros = true )
    const int upstreamImageGlobalOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamImageSizeSquared;

    if( localId < gUpstreamImageSizeSquared ) {
        _upstreamImage[localId] = upstreamImagesGlobal[upstreamImageGlobalOffset + localId];
    }
    int resultImageGlobalOffset = ( n * gNumOutPlanes + outPlane ) * gOutImageSizeSquared;
    if( localId < gOutImageSizeSquared ) {
        _resultImage[localId ] = resultsGlobal[resultImageGlobalOffset + localId];
        _errorImage[localId ] = errorsGlobal[resultImageGlobalOffset + localId];
        _weightReduceArea[localId] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // now we loop over the filter, and the output image...
    for( int filterRow = 0; filterRow < gFilterSize; filterRow++ ) {
//        int outRow = upstreamRow + gMargin - filterRow;
        int upstreamRow = outRow - gMargin + filterRow;
        for( int filterCol = 0; filterCol < gFilterSize; filterCol++ ) {
            int upstreamCol = outCol - gMargin + filterCol;
//            int outCol = upstreamCol + gMargin - filterCol;
//            float thiswchange = 0;
            int resultIndex = outRow * gOutImageSize + outCol;
            float error = _errorImage[resultIndex];
            float actualOutput = _resultImage[resultIndex];
            float activationDerivative = ACTIVATION_DERIV( actualOutput);
            int upstreamDataIndex = upstreamRow * gUpstreamImageSize + upstreamCol;
            float upstreamResult = _upstreamImage[upstreamDataIndex];
            float thisimagethiswchange = upstreamResult * activationDerivative * error;
            if( localId < gOutImageSizeSquared ) {
                _weightReduceArea[localId] = thisimagethiswchange;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            for( int offset = workgroupsizenextpower2 >> 1; offset > 0; offset >>= 1 ) {
                if( localId + offset < gOutImageSizeSquared ) {
                    _weightReduceArea[localId] = _weightReduceArea[ localId ] + _weightReduceArea[ localId + offset ];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if( localId == 0 ) { // maybe can remove this if? leave for now, so fewer bugs :-)
                _weightChanges[filterRow * gFilterSize + filterCol] = _weightReduceArea[0];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // oh, we have to reduce again, over n and stuff...
    // let's test with a single example and upplane and filter first :-)
    // so, we just copy it in for now :-)
    if( localId < gFilterSizeSquared ) {
        weightChangesGlobal[ localId ] = - learningRateMultiplier * _weightChanges[ localId ];
    }

//    if( localId < gUpstreamImageSizeSquared ) {
//        weightChangesGlobal[ globalId ] = upstreamImagesGlobal[globalId];
//        weightChangesGlobal[ globalId ] = _errorImage[globalId];
//    }
//    weightChangesGlobal[globalId] = resultsGlobal[localId];
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
//    weightChanges[ globalId ] = - learningRateMultiplier * thiswchange;
}
*/
#endif
#endif

// need about ~50 workgroups
// eg: - one per n => 128 groups, each one loads:
//                               one cube of input images (46k :-O )
//                               one cube of output images (each output cube is 46k too...)
//                               same errors (46k...)
//                               (need to reduce over n)
//     - one per filter => 32 groups, each one loads:
//                               each cube of input images (eg, sequentially) (each cube is 46k...)
//                               each of its own output planes (eg, sequentially)
//                               each of its own error planes (eg, sequentially)
//                               (no extra-workgroup reduction needed, for imagesize < 19)
//     - one per filter, per upstream => 32*32 = 784 groups. each one loads:
//                               one plane from each cube of images (eg, sequentially) (1.5k per plane)
//                               one plane from each example output (eg, sequentially) (1.5k per plane)
//                               one plane from each example error (eg, sequentially) (1.5k per plane)
//                               each workgroup will have one thread per image position, ie 384 threads
//                               each thread will iterate over the 25 filter positions
//                               after iterating over all n,
//                                   each workgroup will then give a single w update, 5x5 = 100 bytes
//                                    => written to global memory somewhere
//                               and there will be 784 workgroups to reduce over....
//                                   ... but they will be reduced in blocks of 32, giving a cube of 32 filter image
//                                        updates
//                               (and then need to reduce over upstream images)

// if break for per-example, per-filter:
// outimage(eg 28x28), error (28x28), upstreamimage(32x28x28) => weightchanges(32x5x5)
//             784 3k         784 3k                 25088 100k                800 3k
// if break for per-example, per-filter, per-upstream:
// outimage(eg 28x28), error (28x28), upstreamimage(28x28) => weightchanges(5x5)
//             784 3k         784 3k                 784 3k                 25
// n * outplane = 128 * 32 = 4096   , then loop over: [upstreamrow][upstreamcol]
// in this version, workgroups are [outPlane][upstreamPlane]
//                  localid is structured as [upstreamRow][upstreamCol]
//                  globalid is structured as: [outPlane][upstreamPlane]:[upstreamRow][upstreamCol]
//                          each thread should loop over: [n]
//               (and then we will need to reduce each block of 32 filters)
//        (outRow/outCol are locked to upstreamRow/upstreamCol)
// w is [filterRow][filterCol]
// this assumes that filterSizeSquared will fit into one workgroup
//  - which is true for Go-images, but not for MNIST :-P
//      - so we will test with cropped MNIST images, 19x19, same as go images :-)
//
// weightChangesGlobal contains one plane from each of the 784 workgroups
// organized as: [outPlane][upstreamPlan]:[filterRow][filterCol] (colon marks gap between
//                       the coordinates per workgroup, and intra-workgroup coordinates )
#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
#ifdef gOutImageSize // for previous tests that dont define it
/*
void kernel backprop_floats_3( 
    const float learningRateMultiplier, const int batchSize, const int workgroupsizenextpower2,
     global const float *upstreamImagesGlobal, global const float *resultsGlobal, global const float *errorsGlobal,
     global float *weightChangesGlobal,
    local float *_upstreamImage, local float *_resultImage, local float *_errorImage, 
    local float *_weightChanges, local float *_weightReduceArea ) {

        // required (minimum...) sizes for local arrays:
        // upstreamimage: upstreamImageSizeSquared
        // resultimage: outImageSizeSquared
        // errorImage: outImageSizeSquaread
        // weightChanges: filterSizeSquared
        // weightReduceArea: upstreamImageSizeSquared, or workflowSize, to be decided :-)
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupSize = get_local_size(0);
    const int workgroupId = get_group_id(0);

    const int outPlane = workgroupId / gUpstreamNumPlanes;
    const int upstreamPlane = workgroupId % gUpstreamNumPlanes;

    const int outRow = localId / gOutImageSize;
    const int outCol = localId % gOutImageSize;

    // wipe _weightChanges first
    // dont need a barrier, just use the barrier from loading the other planes from global memory
    if( localId < gFilterSizeSquared ) {
        _weightChanges[localId] = 0;
    }

    for( int n = 0; n < batchSize; n++ ) {
        // each localid corresponds to one [upstreamRow][upstreamCol] combination
        // we assume that:
        // filterSize <= upstreamImageSize (reasonable... :-) )
        // outImageSize <= upstreamImageSize (true... unless we have a filter with even size, and padZeros = true )
        const int upstreamImageGlobalOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamImageSizeSquared;
        if( localId < gUpstreamImageSizeSquared ) {
            _upstreamImage[localId] = upstreamImagesGlobal[upstreamImageGlobalOffset + localId];
        }
        const int resultImageGlobalOffset = ( n * gNumOutPlanes + outPlane ) * gOutImageSizeSquared;
        if( localId < gOutImageSizeSquared ) {
            _resultImage[localId ] = resultsGlobal[resultImageGlobalOffset + localId];
            _errorImage[localId ] = errorsGlobal[resultImageGlobalOffset + localId];
            _weightReduceArea[localId] = 0; // note: can probably remove this
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // loaded one upstreamimage, one error plane, one output plane :-)

        // now we loop over the filter, and the output image...
        for( int filterRow = 0; filterRow < gFilterSize; filterRow++ ) {
    //        int outRow = upstreamRow + gMargin - filterRow;
            int upstreamRow = outRow - gMargin + filterRow;
            for( int filterCol = 0; filterCol < gFilterSize; filterCol++ ) {
                int upstreamCol = outCol - gMargin + filterCol;
    //            int outCol = upstreamCol + gMargin - filterCol;
    //            float thiswchange = 0;
                int resultIndex = outRow * gOutImageSize + outCol;
                float error = _errorImage[resultIndex];
                float actualOutput = _resultImage[resultIndex];
                float activationDerivative = ACTIVATION_DERIV( actualOutput);
                int upstreamDataIndex = upstreamRow * gUpstreamImageSize + upstreamCol;
                float upstreamResult = _upstreamImage[upstreamDataIndex];
                float thisimagethiswchange = upstreamResult * activationDerivative * error;
                if( localId < gOutImageSizeSquared ) {
                    _weightReduceArea[localId] = thisimagethiswchange;
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                for( int offset = workgroupsizenextpower2 >> 1; offset > 0; offset >>= 1 ) {
                    if( localId + offset < gOutImageSizeSquared ) {  // cos we're reducing over each position
                                                                     // in the output image, which this workgroup
                                                 // has one thread for each position for
                        _weightReduceArea[localId] = _weightReduceArea[ localId ] + _weightReduceArea[ localId + offset ];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if( localId == 0 ) {
                    _weightChanges[filterRow * gFilterSize + filterCol] += _weightReduceArea[0];
                }
            }
        }
    }
    // now copy our local weightchanges to global memroy
    // each thread copies one element
    // so need a fence :-)
    barrier(CLK_LOCAL_MEM_FENCE);
    const int weightImageGlobalOffset = ( outPlane * gUpstreamNumPlanes + upstreamPlane ) * gFilterSizeSquared;
    if( localId < gFilterSizeSquared ) {
        weightChangesGlobal[weightImageGlobalOffset + localId ] = - learningRateMultiplier * _weightChanges[ localId ];
    }
}
*/
#endif
#endif

#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
#ifdef gOutImageSize // for previous tests that dont define it
// 32 workgroups, one per filter
// globalId: [outPlane]:[upstreamRow][upstreamCol]
//   each thread needs to loop over: [n][upstreamPlane][filterRow][filterCol]
/*
void kernel backprop_floats_4( 
    const float learningRateMultiplier, const int batchSize, const int workgroupsizenextpower2,
     global const float *upstreamImagesGlobal, global const float *resultsGlobal, global const float *errorsGlobal,
     global float *weightChangesGlobal,
    local float *_upstreamImage, local float *_resultImage, local float *_errorImage, 
    local float *_weightChanges, local float *_weightReduceArea ) {

        // required (minimum...) sizes for local arrays:
        // upstreamimage: upstreamImageSizeSquared
        // resultimage: outImageSizeSquared
        // errorImage: outImageSizeSquaread
        // weightChanges: filterSizeSquared
        // weightReduceArea: upstreamImageSizeSquared, or workflowSize, to be decided :-)
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupSize = get_local_size(0);
    const int workgroupId = get_group_id(0);

    const int outPlane = workgroupId;

    const int outRow = localId / gOutImageSize;
    const int outCol = localId % gOutImageSize;

    // wipe _weightChanges first
    // dont need a barrier, just use the barrier from loading the other planes from global memory
    if( localId < gFilterSizeSquared ) {
        _weightChanges[localId] = 0;
    }

    for( int n = 0; n < batchSize; n++ ) {
        const int resultImageGlobalOffset = ( n * gNumOutPlanes + outPlane ) * gOutImageSizeSquared;
        if( localId < gOutImageSizeSquared ) {
            _resultImage[localId ] = resultsGlobal[resultImageGlobalOffset + localId];
            _errorImage[localId ] = errorsGlobal[resultImageGlobalOffset + localId];
//            _weightReduceArea[localId] = 0; // note: can probably remove this
        }
        for( int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++ ) {
            // each localid corresponds to one [upstreamRow][upstreamCol] combination
            // we assume that:
            // filterSize <= upstreamImageSize (reasonable... :-) )
            // outImageSize <= upstreamImageSize (true... unless we have a filter with even size, and padZeros = true )
            const int upstreamImageGlobalOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamImageSizeSquared;
            if( localId < gUpstreamImageSizeSquared ) {
                _upstreamImage[localId] = upstreamImagesGlobal[upstreamImageGlobalOffset + localId];
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // loaded one upstreamimage, one error plane, one output plane :-)

            // now we loop over the filter, and the output image...
            for( int filterRow = 0; filterRow < gFilterSize; filterRow++ ) {
        //        int outRow = upstreamRow + gMargin - filterRow;
                int upstreamRow = outRow - gMargin + filterRow;
                for( int filterCol = 0; filterCol < gFilterSize; filterCol++ ) {
                    int upstreamCol = outCol - gMargin + filterCol;
        //            int outCol = upstreamCol + gMargin - filterCol;
        //            float thiswchange = 0;
                    int resultIndex = outRow * gOutImageSize + outCol;
                    float error = _errorImage[resultIndex];
                    float actualOutput = _resultImage[resultIndex];
                    float activationDerivative = ACTIVATION_DERIV( actualOutput);
                    int upstreamDataIndex = upstreamRow * gUpstreamImageSize + upstreamCol;
                    float upstreamResult = _upstreamImage[upstreamDataIndex];
                    float thisimagethiswchange = upstreamResult * activationDerivative * error;
                    if( localId < gOutImageSizeSquared ) {
                        _weightReduceArea[localId] = thisimagethiswchange;
                    }

                    barrier(CLK_LOCAL_MEM_FENCE);
                    for( int offset = workgroupsizenextpower2 >> 1; offset > 0; offset >>= 1 ) {
                        if( localId + offset < gOutImageSizeSquared ) {  // cos we're reducing over each position
                                                                         // in the output image, which this workgroup
                                                     // has one thread for each position for
                            _weightReduceArea[localId] = _weightReduceArea[ localId ] + _weightReduceArea[ localId + offset ];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if( localId == 0 ) {
                        _weightChanges[upstreamPlane * gFilterSizeSquared + filterRow * gFilterSize + filterCol] += _weightReduceArea[0];
                    }
                }
            }
        }
    }
    // now copy our local weightchanges to global memroy
    // each thread copies one element
    // so need a fence :-)
    barrier(CLK_LOCAL_MEM_FENCE);
    const int weightCubeGlobalOffset = outPlane * gUpstreamNumPlanes * gFilterSizeSquared;
    if( localId < gFilterSizeSquared ) {
        // can flatten this a bit... but probablby not a huge effect. flatten if this kernel is acutllay fast...
        for( int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++ ) {
            int intraCubeOffset = upstreamPlane * gFilterSizeSquared + localId;
            weightChangesGlobal[weightCubeGlobalOffset + intraCubeOffset ] = - learningRateMultiplier * _weightChanges[ intraCubeOffset ];
        }
    }
}
*/
#endif
#endif

// globalid is for: [outPlane][upstreamPlane]:[filterRow][filterCol]
//   workgroup is over [filterRow][filterCol]
// per-thread looping over [n][outRow][outCol]
#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
#ifdef gOutImageSize // for previous tests that dont define it
void kernel backprop_floats_withscratch( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *images, global const float *results, global const float *errors, global float *weightChanges,
        local float *_imageImage, local float *_resultImage, local float *_errorImage
 ) {
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    const int outPlane = workgroupId / gUpstreamNumPlanes;
    const int upstreamPlane = workgroupId % gUpstreamNumPlanes;

    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    float thiswchange = 0;
    for( int n = 0; n < batchSize; n++ ) {
        int upstreamImageGlobalOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamImageSizeSquared;
        // need to fetch the image, but it's bigger than us, so will need to loop...
        int numLoopsForUpstream = ( gUpstreamImageSizeSquared + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numLoopsForUpstream; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gUpstreamImageSizeSquared ) {
                _imageImage[thisOffset] = images[ upstreamImageGlobalOffset + thisOffset ];
            }
        }
        int resultImageGlobalOffset = ( n * gNumOutPlanes + outPlane ) * gOutImageSizeSquared;
        int numLoopsForResults = ( gOutImageSizeSquared + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numLoopsForResults; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gOutImageSizeSquared ) {
                _resultImage[thisOffset ] = ( ACTIVATION_DERIV( results[resultImageGlobalOffset + thisOffset] ) )
                    * errors[resultImageGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int outRow = 0; outRow < gOutImageSize; outRow++ ) {
            int upstreamRow = outRow - gMargin + filterRow;
            for( int outCol = 0; outCol < gOutImageSize; outCol++ ) {
                int upstreamCol = outCol - gMargin + filterCol;
                int resultIndex = outRow * gOutImageSize + outCol;
                float activationDerivative = _resultImage[resultIndex];
                int upstreamDataIndex = upstreamRow * gUpstreamImageSize + upstreamCol;
                float upstreamResult = _imageImage[upstreamDataIndex];
                float thisimagethiswchange = upstreamResult * activationDerivative;
                thiswchange += thisimagethiswchange;
            }
        }
    }
    if( localId < gFilterSizeSquared ) {
        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = - learningRateMultiplier * thiswchange;
//        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = workgroupId;
    }
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
}
#endif
#endif

// biasWeightChanges is: [outplane] (only upstreamplane=0 workgroups need to write it)
#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
#ifdef gOutputImageSize // for previous tests that dont define it
void kernel backprop_floats_withscratch_dobias( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *images, global const float *results, global const float *errors, 
        global float *weights,
        #ifdef BIASED
             global float *biasWeights,
        #endif
        local float *_imageImage, local float *_resultImage, local float *_errorImage
 ) {
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    const int outPlane = workgroupId / gInputPlanes;
    const int upstreamPlane = workgroupId % gInputPlanes;

    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    float thiswchange = 0;
#ifdef BIASED
    float thisbiaschange = 0;
#endif
    for( int n = 0; n < batchSize; n++ ) {
        int upstreamImageGlobalOffset = ( n * gInputPlanes + upstreamPlane ) * gInputImageSizeSquared;
        // need to fetch the image, but it's bigger than us, so will need to loop...
        int numLoopsForUpstream = ( gInputImageSizeSquared + workgroupSize - 1 ) / workgroupSize;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < numLoopsForUpstream; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gInputImageSizeSquared ) {
                _imageImage[thisOffset] = images[ upstreamImageGlobalOffset + thisOffset ];
            }
        }
        int resultImageGlobalOffset = ( n * gNumFilters + outPlane ) * gOutputImageSizeSquared;
        int numLoopsForResults = ( gOutputImageSizeSquared + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numLoopsForResults; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gOutputImageSizeSquared ) {
                _resultImage[thisOffset ] = ( ACTIVATION_DERIV( results[resultImageGlobalOffset + thisOffset] ) )
                    * errors[resultImageGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if( localId < gFilterSizeSquared ) {
            for( int outRow = 0; outRow < gOutputImageSize; outRow++ ) {
                int upstreamRow = outRow - gMargin + filterRow;
                for( int outCol = 0; outCol < gOutputImageSize; outCol++ ) {
                    int upstreamCol = outCol - gMargin + filterCol;
                    bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputImageSize
                        && upstreamCol < gInputImageSize;
                    if( proceed ) {
                        int resultIndex = outRow * gOutputImageSize + outCol;
                        float activationDerivative = _resultImage[resultIndex];
                        int upstreamDataIndex = upstreamRow * gInputImageSize + upstreamCol;
                        float upstreamResult = _imageImage[upstreamDataIndex];
                        float thisimagethiswchange = upstreamResult * activationDerivative;
                        thiswchange += thisimagethiswchange;
    #ifdef BIASED
                        thisbiaschange += activationDerivative;
    #endif
                    }
                }
            }
        }
    }
    if( localId < gFilterSizeSquared ) {
        weights[ workgroupId * gFilterSizeSquared + localId ] += - learningRateMultiplier * thiswchange;
//        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = workgroupId;
    }
#ifdef BIASED
    bool writeBias = upstreamPlane == 0 && localId == 0;
    if( writeBias ) {
        biasWeights[outPlane] += - learningRateMultiplier * thisbiaschange;
    }
#endif
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
}
#endif
#endif

/*
    const int outPlane = globalId;
    float thiswchange = 0;
    for( int n = 0; n < batchSize; n++ ) {
        for( int outRow = 0; outRow < gOutImageSize; outRow++ ) {
            for( int outCol = 0; outCol < gOutImageSize; outCol++ ) {
                int resultIndex = ( ( n * gNumOutPlanes 
                          + outPlane ) * gOutImageSize
                          + outRow ) * gOutImageSize
                          + outCol;
                float thisimagethiswchange = errors[resultIndex] * ACTIVATION_DERIV( results[resultIndex] );
                thiswchange += thisimagethiswchange;
            }
        }
    }
    biasWeightChanges[ globalId ] = - learningMultiplier * thiswchange;
*/


// globalid is for: [outPlane][upstreamPlane][some filterids]:[some filterids][filterRow][filterCol]
//   workgroup is [a filterid][filterRow][filterCol]
// per-thread looping over [n][outRow][outCol]
// eg if a filter is 5x5, which is 25 values, we can fit 20 of these into one workgroup
// or possibly 16 perhaps, all using the same input image data at the same time
#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
#ifdef gOutImageSize // for previous tests that dont define it
void kernel backprop_floats_withscratch_batched( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *images, global const float *results, global const float *errors, global float *weightChanges,
        local float *_imageImage, local float *_resultImage, local float *_errorImage
 ) {
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);

    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

    const int outPlane = workgroupId / gUpstreamNumPlanes;
    const int upstreamPlane = workgroupId % gUpstreamNumPlanes;

    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    float thiswchange = 0;
    for( int n = 0; n < batchSize; n++ ) {
        int upstreamImageGlobalOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamImageSizeSquared;
        // need to fetch the image, but it's bigger than us, so will need to loop...
        int numLoopsForUpstream = ( gUpstreamImageSizeSquared + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numLoopsForUpstream; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gUpstreamImageSizeSquared ) {
                _imageImage[thisOffset] = images[ upstreamImageGlobalOffset + thisOffset ];
            }
        }
        int resultImageGlobalOffset = ( n * gNumOutPlanes + outPlane ) * gOutImageSizeSquared;
        int numLoopsForResults = ( gOutImageSizeSquared + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numLoopsForResults; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gOutImageSizeSquared ) {
                _resultImage[thisOffset ] = ( ACTIVATION_DERIV( results[resultImageGlobalOffset + thisOffset] ) )
                    * errors[resultImageGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int outRow = 0; outRow < gOutImageSize; outRow++ ) {
            int upstreamRow = outRow - gMargin + filterRow;
            for( int outCol = 0; outCol < gOutImageSize; outCol++ ) {
                int upstreamCol = outCol - gMargin + filterCol;
                int resultIndex = outRow * gOutImageSize + outCol;
                float activationDerivative = _resultImage[resultIndex];
                int upstreamDataIndex = upstreamRow * gUpstreamImageSize + upstreamCol;
                float upstreamResult = _imageImage[upstreamDataIndex];
                float thisimagethiswchange = upstreamResult * activationDerivative;
                thiswchange += thisimagethiswchange;
            }
        }
    }
    if( localId < gFilterSizeSquared ) {
        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = - learningRateMultiplier * thiswchange;
//        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = workgroupId;
    }
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
}
#endif
#endif

// doesnt have to be fast, just has to be on gpu is only requirement really...
// make globalid be [outplane]
// and localid is also [outplane]
// so, one workgroup, internally structured as [outplane] (unless there are >512 outplanes....)
#ifdef gOutImageSize // for previous tests that dont define it
kernel void doBiasBackprop( const float learningMultiplier, const int batchSize,
    global float const *results, global float const *errors, global float *biasWeightChanges ) {
    const int globalId = get_local_id(0);
    
    const int outPlane = globalId;

    // bias...
    // biasweights: [outPlane]
    //       aggregate over:  [upstreamPlane][filterRow][filterCol][outRow][outCol][n]
    float thiswchange = 0;
    for( int n = 0; n < batchSize; n++ ) {
        for( int outRow = 0; outRow < gOutImageSize; outRow++ ) {
            for( int outCol = 0; outCol < gOutImageSize; outCol++ ) {
                int resultIndex = ( ( n * gNumOutPlanes 
                          + outPlane ) * gOutImageSize
                          + outRow ) * gOutImageSize
                          + outCol;
                float thisimagethiswchange = errors[resultIndex] * ACTIVATION_DERIV( results[resultIndex] );
                thiswchange += thisimagethiswchange;
            }
        }
    }
    if( globalId < gNumOutPlanes ) {
        biasWeightChanges[ globalId ] = - learningMultiplier * thiswchange;
    }
}
#endif

