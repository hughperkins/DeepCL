// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR | SIGMOID ]
// BIASED (or not)

#ifdef TANH
    #define ACTIVATION_DERIV(output) (1 - output * output)
#elif defined SIGMOID
    #define ACTIVATION_DERIV(output) (output * ( 1 - output ) )
#elif defined RELU
    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : 0)
#elif defined LINEAR
    #define ACTIVATION_DERIV(output) (1.0f)
#endif

// globalid as: [n][upstreamPlane][upstreamrow][upstreamcol]
#ifdef ACTIVATION_DERIV
void kernel calcErrorsForUpstream( 
        const int batchSize,
        global const float *inputData, global const float *errors, global float *weights, global float *errorsForUpstream ) {
    int globalId = get_global_id(0);

    const int upstreamBoard2dId = globalId / gInputBoardSizeSquared;

    const int intraBoardOffset = globalId % gInputBoardSizeSquared;
    const int upstreamRow = intraBoardOffset / gInputBoardSize;
    const int upstreamCol = intraBoardOffset % gInputBoardSize;

    const int upstreamPlane = upstreamBoard2dId % gInputPlanes;
    const int n = upstreamBoard2dId / gInputPlanes;

    const int minFilterRow = max( 0, upstreamRow + gMargin - (gOutputBoardSize - 1) );
    const int maxFilterRow = min( gFilterSize - 1, upstreamRow + gMargin );
    const int minFilterCol = max( 0, upstreamCol + gMargin - (gOutputBoardSize -1) );
    const int maxFilterCol = min( gFilterSize - 1, upstreamCol + gMargin );

    float sumWeightTimesOutError = 0;
    int inputDataIndex = globalId;
    float inputDataValue = inputData[inputDataIndex];
    float inputDeriv = ACTIVATION_DERIV( inputDataValue );
    // aggregate over [outPlane][outRow][outCol]
    for( int outPlane = 0; outPlane < gNumFilters; outPlane++ ) {
        for( int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++ ) {
            int outRow = upstreamRow + gMargin - filterRow;
            for( int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++ ) {
                int outCol = upstreamCol + gMargin - filterCol;
                int resultIndex = ( ( n * gNumFilters 
                          + outPlane ) * gOutputBoardSize
                          + outRow ) * gOutputBoardSize
                          + outCol;
                float thisError = errors[resultIndex];
                int thisWeightIndex = ( ( outPlane * gInputPlanes
                                    + upstreamPlane ) * gFilterSize
                                    + filterRow ) * gFilterSize
                                    + filterCol;
                float thisWeight = weights[thisWeightIndex];
                float thisWeightTimesError = thisWeight * thisError;
                sumWeightTimesOutError += thisWeightTimesError;
            }
        }
    }
    errorsForUpstream[globalId] = sumWeightTimesOutError * inputDeriv;
}
#endif

