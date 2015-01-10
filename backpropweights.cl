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
#elif defined RELU
    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : 0)
#elif defined LINEAR
    #define ACTIVATION_DERIV(output) (1)
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
// outboards(eg 128x32x28x28), errors (eg 128x28x28), upstreamboards (eg 128x32x28x28) => weightchanges (eg 32x32x28x28)
// if break for per-example, per-filter:
// outboard(eg 28x28), error (28x28), upstreamboard(32x28x28) => weightchanges(32x5x5)
//             784 3k         784 3k                 25088 100k                800 3k
// if break for per-filter:
// outboard(eg 128x28x28), error (128x28x28), upstreamboard(128x32x28x28) => weightchanges(32x32x5x5)
//                350k           350k                 12.8MB                   100k
// if break for per-example:
// outboard(eg 28x28), error (28x28), upstreamboard(32x28x28) => weightchanges(32x5x5)
//                3k             3k                 100k                       3k
//    note that weightchanges will need to be summed over 128 input boards
//
// globalid is for: [outPlane][upstreamPlane][filterRow][filterCol]
// per-thread looping over [n][outRow][outCol]
#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
void kernel backprop_floats( const float learningRateMultiplier,
        const int batchSize, const int upstreamNumPlanes, const int numPlanes, 
         const int upstreamBoardSize, const int filterSize, const int outBoardSize, const int padZeros, 
         global const float *images, global const float *results, global const float *errors, 
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
        for( int outRow = 0; outRow < outBoardSize; outRow++ ) {
            int upstreamRow = outRow - margin + filterRow;
            for( int outCol = 0; outCol < outBoardSize; outCol++ ) {
                int upstreamCol = outCol - margin + filterCol;
                int resultIndex = ( ( n * numPlanes 
                          + outPlane ) * outBoardSize
                          + outRow ) * outBoardSize
                          + outCol;
                float error = errors[resultIndex];
                float actualOutput = results[resultIndex];
                float activationDerivative = ACTIVATION_DERIV( actualOutput);
                int upstreamDataIndex = ( ( n * upstreamNumPlanes 
                                 + upstreamPlane ) * upstreamBoardSize
                                 + upstreamRow ) * upstreamBoardSize
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
#ifdef gOutBoardSize // for previous tests that dont define it
/*void kernel backprop_floats_2( 
    const float learningRateMultiplier, const int batchSize, const int workgroupsizenextpower2,
     global const float *upstreamBoardsGlobal, global const float *resultsGlobal, global const float *errorsGlobal,
     global float *weightChangesGlobal,
    local float *_upstreamBoard, local float *_resultBoard, local float *_errorBoard, 
    local float *_weightChanges, local float *_weightReduceArea ) {

        // required (minimum...) sizes for local arrays:
        // upstreamboard: upstreamBoardSizeSquared
        // resultboard: outBoardSizeSquared
        // errorBoard: outBoardSizeSquaread
        // weightChanges: filterSizeSquared
        // weightReduceArea: upstreamBoardSizeSquared, or workflowSize, to be decided :-)
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupSize = get_local_size(0);

    const int upstreamBoard2dId = globalId / gUpstreamBoardSizeSquared;
    const int upstreamPlane = upstreamBoard2dId % gUpstreamNumPlanes;
    const int outPlane2dId = upstreamBoard2dId / gUpstreamNumPlanes;
    const int n = outPlane2dId / gNumOutPlanes;
    const int outPlane = outPlane2dId % gNumOutPlanes;

    const int outRow = localId / gOutBoardSize;
    const int outCol = localId % gOutBoardSize;

    // each localid corresponds to one [upstreamRow][upstreamCol] combination
    // we assume that:
    // filterSize <= upstreamBoardSize (reasonable... :-) )
    // outBoardSize <= upstreamBoardSize (true... unless we have a filter with even size, and padZeros = true )
    const int upstreamBoardGlobalOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;

    if( localId < gUpstreamBoardSizeSquared ) {
        _upstreamBoard[localId] = upstreamBoardsGlobal[upstreamBoardGlobalOffset + localId];
    }
    int resultBoardGlobalOffset = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared;
    if( localId < gOutBoardSizeSquared ) {
        _resultBoard[localId ] = resultsGlobal[resultBoardGlobalOffset + localId];
        _errorBoard[localId ] = errorsGlobal[resultBoardGlobalOffset + localId];
        _weightReduceArea[localId] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // now we loop over the filter, and the output board...
    for( int filterRow = 0; filterRow < gFilterSize; filterRow++ ) {
//        int outRow = upstreamRow + gMargin - filterRow;
        int upstreamRow = outRow - gMargin + filterRow;
        for( int filterCol = 0; filterCol < gFilterSize; filterCol++ ) {
            int upstreamCol = outCol - gMargin + filterCol;
//            int outCol = upstreamCol + gMargin - filterCol;
//            float thiswchange = 0;
            int resultIndex = outRow * gOutBoardSize + outCol;
            float error = _errorBoard[resultIndex];
            float actualOutput = _resultBoard[resultIndex];
            float activationDerivative = ACTIVATION_DERIV( actualOutput);
            int upstreamDataIndex = upstreamRow * gUpstreamBoardSize + upstreamCol;
            float upstreamResult = _upstreamBoard[upstreamDataIndex];
            float thisimagethiswchange = upstreamResult * activationDerivative * error;
            if( localId < gOutBoardSizeSquared ) {
                _weightReduceArea[localId] = thisimagethiswchange;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            for( int offset = workgroupsizenextpower2 >> 1; offset > 0; offset >>= 1 ) {
                if( localId + offset < gOutBoardSizeSquared ) {
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

//    if( localId < gUpstreamBoardSizeSquared ) {
//        weightChangesGlobal[ globalId ] = upstreamBoardsGlobal[globalId];
//        weightChangesGlobal[ globalId ] = _errorBoard[globalId];
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
//                               one cube of input boards (46k :-O )
//                               one cube of output boards (each output cube is 46k too...)
//                               same errors (46k...)
//                               (need to reduce over n)
//     - one per filter => 32 groups, each one loads:
//                               each cube of input boards (eg, sequentially) (each cube is 46k...)
//                               each of its own output planes (eg, sequentially)
//                               each of its own error planes (eg, sequentially)
//                               (no extra-workgroup reduction needed, for boardsize < 19)
//     - one per filter, per upstream => 32*32 = 784 groups. each one loads:
//                               one plane from each cube of boards (eg, sequentially) (1.5k per plane)
//                               one plane from each example output (eg, sequentially) (1.5k per plane)
//                               one plane from each example error (eg, sequentially) (1.5k per plane)
//                               each workgroup will have one thread per board position, ie 384 threads
//                               each thread will iterate over the 25 filter positions
//                               after iterating over all n,
//                                   each workgroup will then give a single w update, 5x5 = 100 bytes
//                                    => written to global memory somewhere
//                               and there will be 784 workgroups to reduce over....
//                                   ... but they will be reduced in blocks of 32, giving a cube of 32 filter board
//                                        updates
//                               (and then need to reduce over upstream boards)

// if break for per-example, per-filter:
// outboard(eg 28x28), error (28x28), upstreamboard(32x28x28) => weightchanges(32x5x5)
//             784 3k         784 3k                 25088 100k                800 3k
// if break for per-example, per-filter, per-upstream:
// outboard(eg 28x28), error (28x28), upstreamboard(28x28) => weightchanges(5x5)
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
//  - which is true for Go-boards, but not for MNIST :-P
//      - so we will test with cropped MNIST images, 19x19, same as go boards :-)
//
// weightChangesGlobal contains one plane from each of the 784 workgroups
// organized as: [outPlane][upstreamPlan]:[filterRow][filterCol] (colon marks gap between
//                       the coordinates per workgroup, and intra-workgroup coordinates )
#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
#ifdef gOutBoardSize // for previous tests that dont define it
/*
void kernel backprop_floats_3( 
    const float learningRateMultiplier, const int batchSize, const int workgroupsizenextpower2,
     global const float *upstreamBoardsGlobal, global const float *resultsGlobal, global const float *errorsGlobal,
     global float *weightChangesGlobal,
    local float *_upstreamBoard, local float *_resultBoard, local float *_errorBoard, 
    local float *_weightChanges, local float *_weightReduceArea ) {

        // required (minimum...) sizes for local arrays:
        // upstreamboard: upstreamBoardSizeSquared
        // resultboard: outBoardSizeSquared
        // errorBoard: outBoardSizeSquaread
        // weightChanges: filterSizeSquared
        // weightReduceArea: upstreamBoardSizeSquared, or workflowSize, to be decided :-)
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupSize = get_local_size(0);
    const int workgroupId = get_group_id(0);

    const int outPlane = workgroupId / gUpstreamNumPlanes;
    const int upstreamPlane = workgroupId % gUpstreamNumPlanes;

    const int outRow = localId / gOutBoardSize;
    const int outCol = localId % gOutBoardSize;

    // wipe _weightChanges first
    // dont need a barrier, just use the barrier from loading the other planes from global memory
    if( localId < gFilterSizeSquared ) {
        _weightChanges[localId] = 0;
    }

    for( int n = 0; n < batchSize; n++ ) {
        // each localid corresponds to one [upstreamRow][upstreamCol] combination
        // we assume that:
        // filterSize <= upstreamBoardSize (reasonable... :-) )
        // outBoardSize <= upstreamBoardSize (true... unless we have a filter with even size, and padZeros = true )
        const int upstreamBoardGlobalOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;
        if( localId < gUpstreamBoardSizeSquared ) {
            _upstreamBoard[localId] = upstreamBoardsGlobal[upstreamBoardGlobalOffset + localId];
        }
        const int resultBoardGlobalOffset = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared;
        if( localId < gOutBoardSizeSquared ) {
            _resultBoard[localId ] = resultsGlobal[resultBoardGlobalOffset + localId];
            _errorBoard[localId ] = errorsGlobal[resultBoardGlobalOffset + localId];
            _weightReduceArea[localId] = 0; // note: can probably remove this
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // loaded one upstreamboard, one error plane, one output plane :-)

        // now we loop over the filter, and the output board...
        for( int filterRow = 0; filterRow < gFilterSize; filterRow++ ) {
    //        int outRow = upstreamRow + gMargin - filterRow;
            int upstreamRow = outRow - gMargin + filterRow;
            for( int filterCol = 0; filterCol < gFilterSize; filterCol++ ) {
                int upstreamCol = outCol - gMargin + filterCol;
    //            int outCol = upstreamCol + gMargin - filterCol;
    //            float thiswchange = 0;
                int resultIndex = outRow * gOutBoardSize + outCol;
                float error = _errorBoard[resultIndex];
                float actualOutput = _resultBoard[resultIndex];
                float activationDerivative = ACTIVATION_DERIV( actualOutput);
                int upstreamDataIndex = upstreamRow * gUpstreamBoardSize + upstreamCol;
                float upstreamResult = _upstreamBoard[upstreamDataIndex];
                float thisimagethiswchange = upstreamResult * activationDerivative * error;
                if( localId < gOutBoardSizeSquared ) {
                    _weightReduceArea[localId] = thisimagethiswchange;
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                for( int offset = workgroupsizenextpower2 >> 1; offset > 0; offset >>= 1 ) {
                    if( localId + offset < gOutBoardSizeSquared ) {  // cos we're reducing over each position
                                                                     // in the output board, which this workgroup
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
    const int weightBoardGlobalOffset = ( outPlane * gUpstreamNumPlanes + upstreamPlane ) * gFilterSizeSquared;
    if( localId < gFilterSizeSquared ) {
        weightChangesGlobal[weightBoardGlobalOffset + localId ] = - learningRateMultiplier * _weightChanges[ localId ];
    }
}
*/
#endif
#endif

#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
#ifdef gOutBoardSize // for previous tests that dont define it
// 32 workgroups, one per filter
// globalId: [outPlane]:[upstreamRow][upstreamCol]
//   each thread needs to loop over: [n][upstreamPlane][filterRow][filterCol]
/*
void kernel backprop_floats_4( 
    const float learningRateMultiplier, const int batchSize, const int workgroupsizenextpower2,
     global const float *upstreamBoardsGlobal, global const float *resultsGlobal, global const float *errorsGlobal,
     global float *weightChangesGlobal,
    local float *_upstreamBoard, local float *_resultBoard, local float *_errorBoard, 
    local float *_weightChanges, local float *_weightReduceArea ) {

        // required (minimum...) sizes for local arrays:
        // upstreamboard: upstreamBoardSizeSquared
        // resultboard: outBoardSizeSquared
        // errorBoard: outBoardSizeSquaread
        // weightChanges: filterSizeSquared
        // weightReduceArea: upstreamBoardSizeSquared, or workflowSize, to be decided :-)
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int workgroupSize = get_local_size(0);
    const int workgroupId = get_group_id(0);

    const int outPlane = workgroupId;

    const int outRow = localId / gOutBoardSize;
    const int outCol = localId % gOutBoardSize;

    // wipe _weightChanges first
    // dont need a barrier, just use the barrier from loading the other planes from global memory
    if( localId < gFilterSizeSquared ) {
        _weightChanges[localId] = 0;
    }

    for( int n = 0; n < batchSize; n++ ) {
        const int resultBoardGlobalOffset = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared;
        if( localId < gOutBoardSizeSquared ) {
            _resultBoard[localId ] = resultsGlobal[resultBoardGlobalOffset + localId];
            _errorBoard[localId ] = errorsGlobal[resultBoardGlobalOffset + localId];
//            _weightReduceArea[localId] = 0; // note: can probably remove this
        }
        for( int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++ ) {
            // each localid corresponds to one [upstreamRow][upstreamCol] combination
            // we assume that:
            // filterSize <= upstreamBoardSize (reasonable... :-) )
            // outBoardSize <= upstreamBoardSize (true... unless we have a filter with even size, and padZeros = true )
            const int upstreamBoardGlobalOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;
            if( localId < gUpstreamBoardSizeSquared ) {
                _upstreamBoard[localId] = upstreamBoardsGlobal[upstreamBoardGlobalOffset + localId];
            }
            barrier(CLK_LOCAL_MEM_FENCE);  // loaded one upstreamboard, one error plane, one output plane :-)

            // now we loop over the filter, and the output board...
            for( int filterRow = 0; filterRow < gFilterSize; filterRow++ ) {
        //        int outRow = upstreamRow + gMargin - filterRow;
                int upstreamRow = outRow - gMargin + filterRow;
                for( int filterCol = 0; filterCol < gFilterSize; filterCol++ ) {
                    int upstreamCol = outCol - gMargin + filterCol;
        //            int outCol = upstreamCol + gMargin - filterCol;
        //            float thiswchange = 0;
                    int resultIndex = outRow * gOutBoardSize + outCol;
                    float error = _errorBoard[resultIndex];
                    float actualOutput = _resultBoard[resultIndex];
                    float activationDerivative = ACTIVATION_DERIV( actualOutput);
                    int upstreamDataIndex = upstreamRow * gUpstreamBoardSize + upstreamCol;
                    float upstreamResult = _upstreamBoard[upstreamDataIndex];
                    float thisimagethiswchange = upstreamResult * activationDerivative * error;
                    if( localId < gOutBoardSizeSquared ) {
                        _weightReduceArea[localId] = thisimagethiswchange;
                    }

                    barrier(CLK_LOCAL_MEM_FENCE);
                    for( int offset = workgroupsizenextpower2 >> 1; offset > 0; offset >>= 1 ) {
                        if( localId + offset < gOutBoardSizeSquared ) {  // cos we're reducing over each position
                                                                         // in the output board, which this workgroup
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
#ifdef gOutBoardSize // for previous tests that dont define it
void kernel backprop_floats_withscratch( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *images, global const float *results, global const float *errors, global float *weightChanges,
        local float *_imageBoard, local float *_resultBoard, local float *_errorBoard
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
        int upstreamBoardGlobalOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;
        // need to fetch the board, but it's bigger than us, so will need to loop...
        int numLoopsForUpstream = ( gUpstreamBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numLoopsForUpstream; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gUpstreamBoardSizeSquared ) {
                _imageBoard[thisOffset] = images[ upstreamBoardGlobalOffset + thisOffset ];
            }
        }
        int resultBoardGlobalOffset = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared;
        int numLoopsForResults = ( gOutBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numLoopsForResults; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gOutBoardSizeSquared ) {
                _resultBoard[thisOffset ] = ( ACTIVATION_DERIV( results[resultBoardGlobalOffset + thisOffset] ) )
                    * errors[resultBoardGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int outRow = 0; outRow < gOutBoardSize; outRow++ ) {
            int upstreamRow = outRow - gMargin + filterRow;
            for( int outCol = 0; outCol < gOutBoardSize; outCol++ ) {
                int upstreamCol = outCol - gMargin + filterCol;
                int resultIndex = outRow * gOutBoardSize + outCol;
                float activationDerivative = _resultBoard[resultIndex];
                int upstreamDataIndex = upstreamRow * gUpstreamBoardSize + upstreamCol;
                float upstreamResult = _imageBoard[upstreamDataIndex];
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
#ifdef gOutputBoardSize // for previous tests that dont define it
void kernel backprop_floats_withscratch_dobias( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *images, global const float *results, global const float *errors, 
        global float *weights,
        #ifdef BIASED
             global float *biasWeights,
        #endif
        local float *_imageBoard, local float *_resultBoard, local float *_errorBoard
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
        int upstreamBoardGlobalOffset = ( n * gInputPlanes + upstreamPlane ) * gInputBoardSizeSquared;
        // need to fetch the board, but it's bigger than us, so will need to loop...
        int numLoopsForUpstream = ( gInputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < numLoopsForUpstream; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gInputBoardSizeSquared ) {
                _imageBoard[thisOffset] = images[ upstreamBoardGlobalOffset + thisOffset ];
            }
        }
        int resultBoardGlobalOffset = ( n * gNumFilters + outPlane ) * gOutputBoardSizeSquared;
        int numLoopsForResults = ( gOutputBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numLoopsForResults; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gOutputBoardSizeSquared ) {
                _resultBoard[thisOffset ] = ( ACTIVATION_DERIV( results[resultBoardGlobalOffset + thisOffset] ) )
                    * errors[resultBoardGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if( localId < gFilterSizeSquared ) {
            for( int outRow = 0; outRow < gOutputBoardSize; outRow++ ) {
                int upstreamRow = outRow - gMargin + filterRow;
                for( int outCol = 0; outCol < gOutputBoardSize; outCol++ ) {
                    int upstreamCol = outCol - gMargin + filterCol;
                    int resultIndex = outRow * gOutputBoardSize + outCol;
                    float activationDerivative = _resultBoard[resultIndex];
                    int upstreamDataIndex = upstreamRow * gInputBoardSize + upstreamCol;
                    float upstreamResult = _imageBoard[upstreamDataIndex];
                    float thisimagethiswchange = upstreamResult * activationDerivative;
                    thiswchange += thisimagethiswchange;
#ifdef BIASED
                    thisbiaschange += activationDerivative;
#endif
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
        for( int outRow = 0; outRow < gOutBoardSize; outRow++ ) {
            for( int outCol = 0; outCol < gOutBoardSize; outCol++ ) {
                int resultIndex = ( ( n * gNumOutPlanes 
                          + outPlane ) * gOutBoardSize
                          + outRow ) * gOutBoardSize
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
// or possibly 16 perhaps, all using the same input board data at the same time
#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
#ifdef gOutBoardSize // for previous tests that dont define it
void kernel backprop_floats_withscratch_batched( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *images, global const float *results, global const float *errors, global float *weightChanges,
        local float *_imageBoard, local float *_resultBoard, local float *_errorBoard
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
        int upstreamBoardGlobalOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;
        // need to fetch the board, but it's bigger than us, so will need to loop...
        int numLoopsForUpstream = ( gUpstreamBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numLoopsForUpstream; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gUpstreamBoardSizeSquared ) {
                _imageBoard[thisOffset] = images[ upstreamBoardGlobalOffset + thisOffset ];
            }
        }
        int resultBoardGlobalOffset = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared;
        int numLoopsForResults = ( gOutBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
        for( int i = 0; i < numLoopsForResults; i++ ) {
            int thisOffset = i * workgroupSize + localId;
            if( thisOffset < gOutBoardSizeSquared ) {
                _resultBoard[thisOffset ] = ( ACTIVATION_DERIV( results[resultBoardGlobalOffset + thisOffset] ) )
                    * errors[resultBoardGlobalOffset + thisOffset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int outRow = 0; outRow < gOutBoardSize; outRow++ ) {
            int upstreamRow = outRow - gMargin + filterRow;
            for( int outCol = 0; outCol < gOutBoardSize; outCol++ ) {
                int upstreamCol = outCol - gMargin + filterCol;
                int resultIndex = outRow * gOutBoardSize + outCol;
                float activationDerivative = _resultBoard[resultIndex];
                int upstreamDataIndex = upstreamRow * gUpstreamBoardSize + upstreamCol;
                float upstreamResult = _imageBoard[upstreamDataIndex];
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
#ifdef gOutBoardSize // for previous tests that dont define it
kernel void doBiasBackprop( const float learningMultiplier, const int batchSize,
    global float const *results, global float const *errors, global float *biasWeightChanges ) {
    const int globalId = get_local_id(0);
    
    const int outPlane = globalId;

    // bias...
    // biasweights: [outPlane]
    //       aggregate over:  [upstreamPlane][filterRow][filterCol][outRow][outCol][n]
    float thiswchange = 0;
    for( int n = 0; n < batchSize; n++ ) {
        for( int outRow = 0; outRow < gOutBoardSize; outRow++ ) {
            for( int outCol = 0; outCol < gOutBoardSize; outCol++ ) {
                int resultIndex = ( ( n * gNumOutPlanes 
                          + outPlane ) * gOutBoardSize
                          + outRow ) * gOutBoardSize
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

