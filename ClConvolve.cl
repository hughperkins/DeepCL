// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR ]
// BIASED (or not)

#ifdef TANH
    #define ACTIVATION_FUNCTION(output) tanh(output)
    #define ACTIVATION_DERIV(output) 1 - output * output
#elif defined RELU
    #define ACTIVATION_FUNCTION(output) output> 0 ? output : 0
    #define ACTIVATION_DERIV(output) output > 0 ? 1 : 0
#elif defined LINEAR
    #define ACTIVATION_FUNCTION(output) output
    #define ACTIVATION_DERIV(output) 1
#endif
    
void kernel convolve_ints( global const int *p_boardSize, global const int *p_filterSize,
      global const int *image, global const int *filter, global int *result ) {
    int id = get_global_id(0);
    int boardSize = p_boardSize[0];
    int filterSize = p_filterSize[0];
    int boardOffset = id / (boardSize * boardSize ) * (boardSize * boardSize );
    int localid = id % (boardSize * boardSize );
    int row = localid / boardSize;
    int col = localid % boardSize;
    int halfFilterSize = filterSize >> 1;
    int sum = 0;
    int minm = max( -halfFilterSize, -row );
    int maxm = min( halfFilterSize, boardSize - 1 - row );
    int minn = max( -halfFilterSize, -col );
    int maxn = min( halfFilterSize, boardSize - 1 - col );
    int m = minm;
    while( m <= maxm ) {
        int x = ( row + m );
        int xboard = boardOffset + x * boardSize;
        int filterrowoffset = (m+halfFilterSize) * filterSize + halfFilterSize;
        int n = minn;
        while( n <= maxn ) {
            int y = col + n;
            sum += image[ xboard + y] * filter[ filterrowoffset + n ];
            n++;
        }
        m++;
    }
    result[id] = sum;
}

void kernel convolve_floats( global const int *p_boardSize, global const int *p_filterSize,
      global const float *image, global const float *filter, global float *result ) {
    int id = get_global_id(0);
    int boardSize = p_boardSize[0];
    int filterSize = p_filterSize[0];
    int boardOffset = id / (boardSize * boardSize ) * (boardSize * boardSize );
    int localid = id % (boardSize * boardSize );
    int row = localid / boardSize;
    int col = localid % boardSize;
    int halfFilterSize = filterSize >> 1;
    float sum = 0;
    int minm = max( -halfFilterSize, -row );
    int maxm = min( halfFilterSize, boardSize - 1 - row );
    int minn = max( -halfFilterSize, -col );
    int maxn = min( halfFilterSize, boardSize - 1 - col );
    int m = minm;
    while( m <= maxm ) {
        int x = ( row + m );
        int xboard = boardOffset + x * boardSize;
        int filterrowoffset = (m+halfFilterSize) * filterSize + halfFilterSize;
        int n = minn;
        while( n <= maxn ) {
            int y = col + n;
            sum += image[ xboard + y] * filter[ filterrowoffset + n ];
            n++;
        }
        m++;
    }
    result[id] = sum;
}

void kernel convolve_imagecubes_int( global const int *p_numInputPlanes, global const int *p_numFilters, 
      global const int *p_boardSize, global const int *p_filterSize,
      global const int *images, global const int *filters, global int *results ) {
    int globalId = get_global_id(0);

    int numInputPlanes = p_numInputPlanes[0];
    int numFilters = p_numFilters[0];
    int boardSize = p_boardSize[0];
    int filterSize = p_filterSize[0];
    int boardSizeSquared = boardSize * boardSize;

    int outputBoard2Id = globalId / boardSizeSquared;
    int filterId = outputBoard2Id % numFilters;
    int inputBoard3Id = outputBoard2Id / numFilters;

    int filterOffset = filterId * filterSize * filterSize;
    int inputBoard3Offset = inputBoard3Id * numInputPlanes * boardSizeSquared;

    // intraboard coords
    int localid = globalId % boardSizeSquared;
    int row = localid / boardSize;
    int col = localid % boardSize;

    int halfFilterSize = filterSize >> 1;
    int sum = 0;
    int minm = max( -halfFilterSize, -row );
    int maxm = min( halfFilterSize, boardSize - 1 - row );
    int minn = max( -halfFilterSize, -col );
    int maxn = min( halfFilterSize, boardSize - 1 - col );
    int plane = 0;
    while( plane < numInputPlanes ) {
        int inputBoardOffset = inputBoard3Offset + plane * boardSizeSquared;
        int filterPlaneOffset = filterOffset + plane * filterSize * filterSize;
        int m = minm;
        while( m <= maxm ) {
            int y = row + m;
            int inputboardrowoffset = inputBoardOffset + y * boardSize;
            int filterrowoffset = filterPlaneOffset + (m+halfFilterSize) * filterSize + halfFilterSize;
            int n = minn;
            while( n <= maxn ) {
                int x = col + n;
                sum += images[ inputboardrowoffset + x] * filters[ filterrowoffset + n ];
                n++;
            }
            m++;
        }
        plane++;
    }
    results[globalId] = sum;
}

// receive images as a stack of images
// globalid = n * numfilters * boardsize * boardsize + filter * boardsize * boardsize + imagerow * boardsize + imagecol
//                                 globalid              globalid
//  inputboard3 1 inputboard2 1----filter 1             -> outputboard2 1   outputboard3 1
//                inputboard2 2_/\_filter 2             -> outputboard2 2
//  inputboard3 2 inputboard2 3    filter 1             -> outputboard2 3   outputboard3 2
//                inputboard2 4    filter 2             -> outputboard2 4
//
// each outputboard is only written once, by a combination of:
// - one inputboard3
// - one filter
// each inputboard3 is mapped to each filter once, each time writing to one outputboard
//
// images is:
//       numimages * numinputplanes * boardsizesquared
// filters is:
//       numfilters * numinputplanes * filtersizesquared
// outputs is:
//       numimages * numfilters * outputboardsizesquared

// images are organized like [imageId][plane][row][col]
// filters are organized like [filterid][plane][filterrow][filtercol]
// results are organized like [imageid][filterid][row][col]
void kernel convolve_imagecubes_float( 
      const int numInputPlanes, const int numFilters, 
      const int boardSize, const int filterSize,
      global const float *images, global const float *filters, global float *results ) {
    int globalId = get_global_id(0);

    int boardSizeSquared = boardSize * boardSize;

    int outputBoard2Id = globalId / boardSizeSquared;
    int filterId = outputBoard2Id % numFilters;
    int inputBoard3Id = outputBoard2Id / numFilters;

    int filterOffset = filterId * filterSize * filterSize;
    int inputBoard3Offset = inputBoard3Id * numInputPlanes * boardSizeSquared;

    // intraboard coords
    int localid = globalId % boardSizeSquared;
    int row = localid / boardSize;
    int col = localid % boardSize;

    int halfFilterSize = filterSize >> 1;
    float sum = 0;
    // m should vary from -halfFilterSize through 0 to halfFilterSize 
    // n too...
    int minm = max( -halfFilterSize, -row );
    int maxm = min( halfFilterSize, boardSize - 1 - row );
    int minn = max( -halfFilterSize, -col );
    int maxn = min( halfFilterSize, boardSize - 1 - col );
    int inputPlane = 0;
    while( inputPlane < numInputPlanes ) {
        int inputBoardOffset = inputBoard3Offset + inputPlane * boardSizeSquared;
        int m = minm;
        while( m <= maxm ) {
            int y = row + m;
            int inputboardrowoffset = inputBoardOffset + y * boardSize;
            int filterrowoffset = filterOffset + (m+halfFilterSize) * filterSize + halfFilterSize;
            int n = minn;
            while( n <= maxn ) {
                int x = col + n;
                sum += images[ inputboardrowoffset + x] * filters[ filterrowoffset + n ];
                n++;
            }
            m++;
        }
        inputPlane++;
    }

    results[globalId] = sum;
}

void kernel convolve_imagecubes_float_nopadzeros( 
      const int numInputPlanes, const int numFilters, 
      const int inputBoardSize, const int filterSize,
      global const float *images, global const float *filters, global float *results ) {
    int globalId = get_global_id(0);

    int inputBoardSizeSquared = inputBoardSize * inputBoardSize;
    int outputBoardSize = inputBoardSize - filterSize + 1;
    int outputBoardSizeSquared = outputBoardSize * outputBoardSize;

    int outputBoard2Id = globalId / outputBoardSizeSquared;
    int filterId = outputBoard2Id % numFilters;
    int inputBoard3Id = outputBoard2Id / numFilters;

    int filterOffset = filterId * filterSize * filterSize;
    int inputBoard3Offset = inputBoard3Id * numInputPlanes * inputBoardSizeSquared;

    // intraboard coords
    int localid = globalId % outputBoardSizeSquared;
    int outputRow = localid / outputBoardSize;
    int outputCol = localid % outputBoardSize;

    int halfFilterSize = filterSize >> 1;
    float sum = 0;
    int minm = -halfFilterSize;
    int maxm = halfFilterSize;
    int minn = -halfFilterSize;
    int maxn = halfFilterSize;
    int inputPlane = 0;
    while( inputPlane < numInputPlanes ) {
        int inputBoardOffset = inputBoard3Offset + inputPlane * inputBoardSizeSquared;
        int m = minm;
        while( m <= maxm ) {
            int inputRow = outputRow + m + halfFilterSize;
            int inputboardrowoffset = inputBoardOffset + inputRow * inputBoardSize;
            int filterrowoffset = filterOffset + (m+halfFilterSize) * filterSize + halfFilterSize;
            int n = minn;
            while( n <= maxn ) {
                int inputCol = outputCol + n + halfFilterSize;
                sum += images[ inputboardrowoffset + inputCol] * filters[ filterrowoffset + n ];
                n++;
            }
            m++;
        }
        inputPlane++;
    }
    results[globalId] = sum;
}

// images are organized like [imageId][plane][row][col]
// filters are organized like [filterid][inplane][filterrow][filtercol]
// results are organized like [imageid][filterid][row][col]
// global id is organized like results, ie: [imageid][filterid][row][col]
#ifdef ACTIVATION_FUNCTION // protect against not defined
void kernel convolve_imagecubes_float2( 
      const int numInputPlanes, const int numFilters, 
      const int inputBoardSize, const int filterSize, const int padZeros,
      global const float *images, global const float *filters, 
#ifdef BIASED
global const float*biases, 
#endif
    global float *results ) {
    int globalId = get_global_id(0);

    int inputBoardSizeSquared = inputBoardSize * inputBoardSize;
    int outputBoardSize = padZeros ? inputBoardSize : inputBoardSize - filterSize + 1;
    int outputBoardSizeSquared = outputBoardSize * outputBoardSize;
    int filterSizeSquared = filterSize * filterSize;

    int outputBoard2Id = globalId / outputBoardSizeSquared;
    int exampleId = outputBoard2Id / numFilters;
    int filterId = outputBoard2Id % numFilters;

    int inputCubeOffset = exampleId * numInputPlanes * inputBoardSizeSquared;
    int filterCubeOffset = filterId * numInputPlanes * filterSizeSquared;

    // intraboard coords
    int localid = globalId % outputBoardSizeSquared;
    int outputRow = localid / outputBoardSize;
    int outputCol = localid % outputBoardSize;

    int halfFilterSize = filterSize >> 1;
    float sum = 0;
    int minm = padZeros ? max( -halfFilterSize, -outputRow ) : -halfFilterSize;
    int maxm = padZeros ? min( halfFilterSize, outputBoardSize - 1 - outputRow ) : halfFilterSize;
    int minn = padZeros ? max( -halfFilterSize, -outputCol ) : - halfFilterSize;
    int maxn = padZeros ? min( halfFilterSize, outputBoardSize - 1 - outputCol ) : halfFilterSize;
    int inputPlane = 0;
    float probe = 0;
    while( inputPlane < numInputPlanes ) {
        int inputBoardOffset = inputCubeOffset + inputPlane * inputBoardSizeSquared;
        int filterBoardOffset = filterCubeOffset + inputPlane * filterSizeSquared;
        int m = minm;
        while( m <= maxm ) {
            int inputRow = outputRow + m + ( padZeros ? 0 : halfFilterSize );
            int inputboardrowoffset = inputBoardOffset + inputRow * inputBoardSize;
            int filterrowoffset = filterBoardOffset + (m+halfFilterSize) * filterSize + halfFilterSize;
            int n = minn;
            while( n <= maxn ) {
                int inputCol = outputCol + n + ( padZeros ? 0 : halfFilterSize );
                sum += images[ inputboardrowoffset + inputCol] * filters[ filterrowoffset + n ];
                //probe += 10000 * pown(100, inputPlane) *( inputboardrowoffset + inputCol );
//                probe += pown(100, inputPlane) *( images[inputboardrowoffset + inputCol] );
                //probe += pown(100, inputPlane) *( filterrowoffset + n );
             //   probe += pown(1000, inputPlane) *( floor(filters[ filterrowoffset + n ]*100)/100 );

//                sum = filters[filterrowoffset + n];
                //sum = filterrowoffset;
                n++;
            }
            m++;
        }
//        probe += pown(100, inputPlane ) * filterBoardOffset;
        inputPlane++;
    }
//     probe = exampleId * 100 + filterCubeOffset;

#ifdef BIASED
    sum += biases[filterId];
#endif
    results[globalId] = ACTIVATION_FUNCTION(sum);
//    results[0] = 1234.0;
     //results[globalId] = probe;
}
#endif

// images are organized like [imageId][plane][row][col]    128*32*19*19=1,500,000
// filters are organized like [filterid][inplane][filterrow][filtercol] 32*32*5*5=25600
// results are organized like [imageid][filterid][row][col]   128*32*19*19=1,500,000
//                  if w updates are per image,then 25600*128 = 3.3 million
// globalid is for: [outPlane][upstreamPlane][filterRow][filterCol]
// eg 32 * 32 * 5 * 5 = 25600 ...
// then we are aggregating over [outRow][outCol][n]
//      eg 19 * 19 * 128 = 46208
// derivtype: 0=relu 1=tanh
#ifdef ACTIVATION_DERIV // protect against if activation_function not defined
void kernel backprop_floats( const float learningRateMultiplier,
        const int batchSize, const int upstreamNumPlanes, const int numPlanes, 
         const int upstreamBoardSize, const int filterSize, const int outBoardSize, const int padZeros, 
         global const float *images, global const float *results, global const float *errors, global float *weightChanges ) {
    int globalId = get_global_id(0);

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
    int outRow = 0;
    while( outRow < outBoardSize ) {
        int upstreamRow = outRow - margin + filterRow;
        int outCol = 0;
        while( outCol < outBoardSize ) {
            int upstreamCol = outCol - margin + filterCol;
            int n = 0;
            while( n < batchSize ) {
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
                n++;
            }
            outCol++;
        }
        outRow++;
    }
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    weightChanges[ globalId ] = - learningRateMultiplier * thiswchange;
}
#endif

//void kernel byelement_add_inplace( global float *target, global const float *src ) {
//    int globalId = get_global_id(0);
//    target[globalId] += src[globalId];
//}

// results are organized like [imageid][filterid][row][col]
// bias si applied per filterId
// need to know:
//  - how many filters
//  - filterSizeSquared
//void kernel apply_bias( const int numFilters, const int filterSizeSquared, global float *target, global const float *biasWeights ) {
//    int globalId = get_global_id(0);
//    int filterId = ( globalId / filterSizeSquared ) % numFilters;
//    target[globalId] += biasWeights[filterId];
//}

//void kernel byelement_mult_inplace( global float *target, const float scalar ) {
//    int globalId = get_global_id(0);
//    target[globalId] *= scalar;
//}

//void kernel byelement_tanh( global float *vector ) {
//    int globalId = get_global_id(0);
//    vector[globalId] = tanh(vector[globalId]);
//}

//void kernel byelement_relu( global float *vector ) {
//    int globalId = get_global_id(0);
//    vector[globalId] = vector[globalId] > 0 ? vector[globalId] : 0;
//}


