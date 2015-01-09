// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR ]
// BIASED (or not)

#ifdef TANH
    #define ACTIVATION_FUNCTION(output) (tanh(output))
    #define ACTIVATION_DERIV(output) (1 - output * output)
#elif defined RELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : 0)
#elif defined LINEAR
    #define ACTIVATION_FUNCTION(output) (output)
    #define ACTIVATION_DERIV(output) (1)
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

// notes on non-odd filtersizes:
// for odd, boardsize and filtersize 3, padZeros = 0:
// output is a single square
// m and n should vary between -1,0,1
// for even, boardsize and filtersize 2, padzeros = 0
// output is a single square, which we can position at topleft or bottomrigth
// lets position it in bottomright
// then m and n should vary as -1,0
//
// for even, boardsize and filtersize 2, padzeros = 1
// output is 2 by 2
// well... if it is even:
// - if we are not padding zeros, then we simply move our filter around the board somehow
// - if we are padding zeros, then we conceptually pad the bottom and right edge of the board with zeros by 1
// filtersize remains the same
//      m will vary as -1,0,1
//       outputrow is fixed by globalid
//       inputrow should be unchanged...
// padzeros = 0:
//  x x .  . . .
//  x x .  . x x
//  . . .  . x x
// when filtersize even:
//    new boardsize = oldboardsize - filtersize + 1
// when filtersize odd:
//    x x x .
//    x x x .
//    x x x .
//    . . . .
//    new boardsize = oldboardsize - filtersize + 1
// padzeros = 1:
// x x
// x x . .   x x .    . . .     . . .
//   . . .   x x .    . x x     . . .
//   . . .   . . .    . x x     . . x x
// outrow=0 outrow=1  outrow=2      x x
// outcol=0 outcol=1  outcol=2    outrow=3
//                                outcol=3
// when filtersize is even, and padzeros, boardsize grows by 1 each time...
//    boardsize = oldboardsize + 1
// when filtersize is odd
//  x x x 
//  x x x .   x x x    . . .
//  x x x .   x x x    . x x x
//    . . .   x x x    . x x x
//                       x x x

// images are organized like [imageId][plane][row][col]
// filters are organized like [filterid][inplane][filterrow][filtercol]
// results are organized like [imageid][filterid][row][col]
// global id is organized like results, ie: [imageid][outplane][outrow][outcol]
// - no local memory used currently
// - each thread:
//     - loads a whole upstream cube
//     - loads a whole filter cube
//     - writes one output...
#ifdef ACTIVATION_FUNCTION // protect against not defined
void kernel convolve_imagecubes_float2( const int numExamples,
      const int numInputPlanes, const int numFilters, 
      const int inputBoardSize, const int filterSize, const int padZeros,
      global const float *images, global const float *filters, 
#ifdef BIASED
global const float*biases, 
#endif
    global float *results ) {
    int globalId = get_global_id(0);

    const int evenPadding = filterSize % 2 == 0 ? 1 : 0;

    int inputBoardSizeSquared = inputBoardSize * inputBoardSize;
    int outputBoardSize = padZeros ? inputBoardSize + evenPadding : inputBoardSize - filterSize + 1;
    int outputBoardSizeSquared = outputBoardSize * outputBoardSize;
    int filterSizeSquared = filterSize * filterSize;

    int outputBoard2Id = globalId / outputBoardSizeSquared;
    int exampleId = outputBoard2Id / numFilters;
    int filterId = outputBoard2Id % numFilters;

    if( exampleId >= numExamples ) {
        return;
    }

    int inputCubeOffset = exampleId * numInputPlanes * inputBoardSizeSquared;
    int filterCubeOffset = filterId * numInputPlanes * filterSizeSquared;

    // intraboard coords
    int localid = globalId % outputBoardSizeSquared;
    int outputRow = localid / outputBoardSize;
    int outputCol = localid % outputBoardSize;

    int halfFilterSize = filterSize >> 1;
    float sum = 0;
    //  boardsize = oldboardsize
    int minm = padZeros ? max( -halfFilterSize, -outputRow ) : -halfFilterSize;
    int maxm = padZeros ? min( halfFilterSize - evenPadding, outputBoardSize - 1 - outputRow  - evenPadding) : halfFilterSize - evenPadding;
    int minn = padZeros ? max( -halfFilterSize, -outputCol ) : - halfFilterSize;
    int maxn = padZeros ? min( halfFilterSize - evenPadding, outputBoardSize - 1 - outputCol - evenPadding) : halfFilterSize - evenPadding;
    int inputPlane = 0;
//    float probe = 0;
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
//                probe += 10000 * pown(100, inputPlane) *( inputboardrowoffset + inputCol );
            //    probe += pown(100, inputPlane) *( images[inputboardrowoffset + inputCol] );
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

//    results[globalId] = globalId;
//    results[0] = 1234.0;
//     results[1024+globalId] = maxn;
//     results[1] = maxMm;
//     results[2] = minm;
}
#endif

#ifdef gOutBoardSize // for previous tests that dont define it
#ifdef ACTIVATION_FUNCTION // protect against not defined
// workgroup id organized like: [outplane]
// local id organized like: [outrow][outcol]
// each thread iterates over: [imageid][upstreamplane][filterrow][filtercol]
// number workgroups = 32
// one filter plane takes up 5 * 5 * 4 = 100 bytes
// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)
// all filter cubes = 3.2KB * 32 = 102KB (too big)
// results are organized like [imageid][filterid][row][col]
void kernel convolve_imagecubes_float3( const int batchSize,
      global const float *images, global const float *filters, 
        #ifdef BIASED
            global const float*biases, 
        #endif
    global float *results,
    local float *_upstreamBoard, local float *_filterCube ) {
    const int globalId = get_global_id(0);

    const int evenPadding = gFilterSize % 2 == 0 ? 1 : 0;

    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int outPlane = workgroupId;

    const int localId = get_local_id(0);
    const int outputRow = localId / gOutBoardSize;
    const int outputCol = localId % gOutBoardSize;

    const int minu = gPadZeros ? max( -gHalfFilterSize, -outputRow ) : -gHalfFilterSize;
    const int maxu = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutBoardSize - 1 - outputRow  - evenPadding) : gHalfFilterSize - evenPadding;
    const int minv = gPadZeros ? max( -gHalfFilterSize, -outputCol ) : - gHalfFilterSize;
    const int maxv = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutBoardSize - 1 - outputCol - evenPadding) : gHalfFilterSize - evenPadding;

    const int numUpstreamsPerThread = ( gUpstreamBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;

    const int filterCubeLength = gUpstreamNumPlanes * gFilterSizeSquared;
    const int filterCubeGlobalOffset = outPlane * filterCubeLength;
    const int numPixelsPerThread = ( filterCubeLength + workgroupSize - 1 ) / workgroupSize;
    for( int i = 0; i < numPixelsPerThread; i++ ) {
        int thisOffset = localId + i * workgroupSize;
        if( thisOffset < filterCubeLength ) {
            _filterCube[thisOffset] = filters[filterCubeGlobalOffset + thisOffset];
        }
    }
    // dont need a barrier, since we'll just run behind the barrier from the upstream board download

    for( int n = 0; n < batchSize; n++ ) {
        float sum = 0;
        for( int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++ ) {
            int thisUpstreamBoardOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;
            barrier(CLK_LOCAL_MEM_FENCE);
            for( int i = 0; i < numUpstreamsPerThread; i++ ) {
                int thisOffset = workgroupSize * i + localId;
                if( thisOffset < gUpstreamBoardSizeSquared ) {
                    _upstreamBoard[ thisOffset ] = images[ thisUpstreamBoardOffset + thisOffset ];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            int filterBoardOffset = upstreamPlane * gFilterSizeSquared;
            if( localId >= gOutBoardSizeSquared ) {
                continue;
            }
            for( int u = minu; u <= maxu; u++ ) {
                int inputRow = outputRow + u + ( gPadZeros ? 0 : gHalfFilterSize );
                int inputboardrowoffset = inputRow * gUpstreamBoardSize;
                int filterrowoffset = filterBoardOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                for( int v = minv; v <= maxv; v++ ) {
                    int inputCol = outputCol + v + ( gPadZeros ? 0 : gHalfFilterSize );
                    sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                }
            }
        }
        #ifdef BIASED
            sum += biases[outPlane];
        #endif
        // results are organized like [imageid][filterid][row][col]
        int resultIndex = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared + localId;
        if( localId < gOutBoardSizeSquared ) {
            results[resultIndex ] = ACTIVATION_FUNCTION(sum);
        }
    }
}
#endif
#endif


#ifdef gOutBoardSize // for previous tests that dont define it
#ifdef ACTIVATION_FUNCTION // protect against not defined
// workgroup id organized like: [imageid][outplane]
// local id organized like: [outrow][outcol]
// each thread iterates over: [upstreamplane][filterrow][filtercol]
// number workgroups = 32
// one filter plane takes up 5 * 5 * 4 = 100 bytes
// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)
// all filter cubes = 3.2KB * 32 = 102KB (too big)
// results are organized like [imageid][filterid][row][col]
void kernel convolve_imagecubes_float4( const int batchSize,
      global const float *images, global const float *filters, 
        #ifdef BIASED
            global const float*biases, 
        #endif
    global float *results,
    local float *_upstreamBoard, local float *_filterCube ) {
    const int globalId = get_global_id(0);

    const int evenPadding = gFilterSize % 2 == 0 ? 1 : 0;

    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int n = workgroupId / gNumOutPlanes;
    const int outPlane = workgroupId % gNumOutPlanes;

    const int localId = get_local_id(0);
    const int outputRow = localId / gOutBoardSize;
    const int outputCol = localId % gOutBoardSize;

    const int minu = gPadZeros ? max( -gHalfFilterSize, -outputRow ) : -gHalfFilterSize;
    const int maxu = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutBoardSize - 1 - outputRow  - evenPadding) : gHalfFilterSize - evenPadding;
    const int minv = gPadZeros ? max( -gHalfFilterSize, -outputCol ) : - gHalfFilterSize;
    const int maxv = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutBoardSize - 1 - outputCol - evenPadding) : gHalfFilterSize - evenPadding;

    const int numUpstreamsPerThread = ( gUpstreamBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;

    const int filterCubeLength = gUpstreamNumPlanes * gFilterSizeSquared;
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
    for( int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++ ) {
        int thisUpstreamBoardOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < numUpstreamsPerThread; i++ ) {
            int thisOffset = workgroupSize * i + localId;
            if( thisOffset < gUpstreamBoardSizeSquared ) {
                _upstreamBoard[ thisOffset ] = images[ thisUpstreamBoardOffset + thisOffset ];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int filterBoardOffset = upstreamPlane * gFilterSizeSquared;
        if( localId >= gOutBoardSizeSquared ) {
            continue;
        }
        for( int u = minu; u <= maxu; u++ ) {
            int inputRow = outputRow + u + ( gPadZeros ? 0 : gHalfFilterSize );
            int inputboardrowoffset = inputRow * gUpstreamBoardSize;
            int filterrowoffset = filterBoardOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for( int v = minv; v <= maxv; v++ ) {
                int inputCol = outputCol + v + ( gPadZeros ? 0 : gHalfFilterSize );
                sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
            }
        }
    }
    #ifdef BIASED
        sum += biases[outPlane];
    #endif
    // results are organized like [imageid][filterid][row][col]
    int resultIndex = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared + localId;
    if( localId < gOutBoardSizeSquared ) {
        results[resultIndex ] = ACTIVATION_FUNCTION(sum);
    }

}
#endif
#endif

#ifdef gOutBoardSize // for previous tests that dont define it
#ifdef ACTIVATION_FUNCTION // protect against not defined
// workgroup id organized like: [imageid][outplane]
// local id organized like: [outrow][outcol]
// each thread iterates over: [upstreamplane][filterrow][filtercol]
// number workgroups = 32
// one filter plane takes up 5 * 5 * 4 = 100 bytes
// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)
// all filter cubes = 3.2KB * 32 = 102KB (too big)
// results are organized like [imageid][filterid][row][col]
void kernel convolve_imagecubes_float5( const int batchSize,
      global const float *images, global const float *filters, 
        #ifdef BIASED
            global const float*biases, 
        #endif
    global float *results,
    local float *_upstreamBoard, local float *_filterCube ) {
    const int globalId = get_global_id(0);

    const int evenPadding = gFilterSize % 2 == 0 ? 1 : 0;

    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int n = workgroupId / gNumOutPlanes;
    const int outPlane = workgroupId % gNumOutPlanes;

    const int localId = get_local_id(0);
    const int outputRow = localId / gOutBoardSize;
    const int outputCol = localId % gOutBoardSize;

    const int minu = gPadZeros ? max( -gHalfFilterSize, -outputRow ) : -gHalfFilterSize;
    const int maxu = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutBoardSize - 1 - outputRow  - evenPadding) : gHalfFilterSize - evenPadding;
    const int minv = gPadZeros ? max( -gHalfFilterSize, -outputCol ) : - gHalfFilterSize;
    const int maxv = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutBoardSize - 1 - outputCol - evenPadding) : gHalfFilterSize - evenPadding;

    const int numUpstreamsPerThread = ( gUpstreamBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
    const int numFilterPixelsPerThread = ( gFilterSizeSquared + workgroupSize - 1 ) / workgroupSize;

    float sum = 0;
    for( int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++ ) {
        int thisUpstreamBoardOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for( int i = 0; i < numUpstreamsPerThread; i++ ) {
            int thisOffset = workgroupSize * i + localId;
            if( thisOffset < gUpstreamBoardSizeSquared ) {
                _upstreamBoard[ thisOffset ] = images[ thisUpstreamBoardOffset + thisOffset ];
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
        if( localId < gOutBoardSizeSquared ) {
            for( int u = minu; u <= maxu; u++ ) {
                int inputRow = outputRow + u + ( gPadZeros ? 0 : gHalfFilterSize );
                int inputboardrowoffset = inputRow * gUpstreamBoardSize;
                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                for( int v = minv; v <= maxv; v++ ) {
                    int inputCol = outputCol + v + ( gPadZeros ? 0 : gHalfFilterSize );
                    sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                }
            }
        }
    }
    #ifdef BIASED
        sum += biases[outPlane];
    #endif
    // results are organized like [imageid][filterid][row][col]
    int resultIndex = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared + localId;
    if( localId < gOutBoardSizeSquared ) {
        results[resultIndex ] = ACTIVATION_FUNCTION(sum);
//        results[resultIndex ] = 123;
    }
}
#endif
#endif

////    if( globalId == 0 ) {
////        for( int i = 0; i < 4; i++ ) {
////            results[14 + i] = thisOffset;
////        }
////    }
//    if( globalId == 0 ) {
//        for( int i = 0; i < gUpstreamBoardSizeSquared; i++ ) {
//            results[100 * (1+upstreamPlane) + i] = _upstreamBoard[i];
//        }
//    }
//    if( globalId == 12 ) {
////        results[400 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] = sum;
////        results[400 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] = _upstreamBoard[ inputboardrowoffset + inputCol];
//        results[400 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] = inputboardrowoffset + inputCol;
////        results[400 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] = minu;
////        results[400 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] += 1;
//        results[600 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] = _filterCube[ filterrowoffset + v ];
//    }
//    if( globalId == 0 ) {
//        for( int i = 0; i < filterCubeLength; i++ ) {
//            results[300 + i] = _filterCube[i];
//        }
//    }
////    if( globalId == 12 ) {
////        results[500 + 0] = 
////    }

////    results[globalId*2] = images[25+globalId];
////    results[globalId*2+1] = _upstreamBoard[globalId];


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
            }
        }
    }
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    weightChanges[ globalId ] = - learningRateMultiplier * thiswchange;
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
#ifdef gOutBoardSize // for previous tests that dont define it
void kernel backprop_floats_withscratch_dobias( 
        const float learningRateMultiplier, const int batchSize, 
         global const float *images, global const float *results, global const float *errors, global float *weightChanges, global float *biasWeightChanges,
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
    float thisbiaschange = 0;
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
                thisbiaschange += activationDerivative;
            }
        }
    }
    if( localId < gFilterSizeSquared ) {
        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = - learningRateMultiplier * thiswchange;
//        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = workgroupId;
    }
    bool writeBias = upstreamPlane == 0 && localId == 0;
    if( writeBias ) {
        biasWeightChanges[outPlane] = - learningRateMultiplier * thisbiaschange;
    }
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

// handle lower layer...
// errors for upstream look like [n][inPlane][inRow][inCol]
// need to aggregate over: [outPlane][outRow][outCol] (?)
// need to backprop errors along each possible weight
// each upstream feeds to:
//    - each of our filters (so numPlanes filters)
//    - each of our outpoint points (so boardSize * boardSize)
// errors are provider per [n][inPlane][inRow][inCol]
// globalid is structured as: [n][upstreamPlane][upstreamRow][upstreamCol]
// there will be approx 128 * 32 * 28 * 28 = 3 million threads :-P
// grouped into 4608 workgroups
// maybe we want fewer than this?
void kernel calcErrorsForUpstream( 
        const int upstreamNumPlanes, const int upstreamBoardSize, const int filterSize, 
        const int outNumPlanes, const int outBoardSize,
        const int padZeros,
        global const float *weights, global const float *errors, global float *errorsForUpstream ) {
    int globalId = get_global_id(0);
    const int halfFilterSize = filterSize >> 1;
    const int margin = padZeros ? halfFilterSize : 0;

    const int upstreamBoardSizeSquared = upstreamBoardSize * upstreamBoardSize;
    const int upstreamBoard2dId = globalId / upstreamBoardSizeSquared;

    const int intraBoardOffset = globalId % upstreamBoardSizeSquared;
    const int upstreamRow = intraBoardOffset / upstreamBoardSize;
    const int upstreamCol = intraBoardOffset % upstreamBoardSize;

    const int upstreamPlane = upstreamBoard2dId % upstreamNumPlanes;
    const int n = upstreamBoard2dId / upstreamNumPlanes;

    const int minFilterRow = max( 0, upstreamRow + margin - (outBoardSize - 1) );
    const int maxFilterRow = min( filterSize - 1, upstreamRow + margin );
    const int minFilterCol = max( 0, upstreamCol + margin - (outBoardSize -1) );
    const int maxFilterCol = min( filterSize - 1, upstreamCol + margin );

    float sumWeightTimesOutError = 0;
    // aggregate over [outPlane][outRow][outCol]
    for( int outPlane = 0; outPlane < outNumPlanes; outPlane++ ) {
        for( int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++ ) {
            int outRow = upstreamRow + margin - filterRow;
            for( int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++ ) {
                int outCol = upstreamCol + margin - filterCol;
                int resultIndex = ( ( n * outNumPlanes 
                          + outPlane ) * outBoardSize
                          + outRow ) * outBoardSize
                          + outCol;
                float thisError = errors[resultIndex];
                int thisWeightIndex = ( ( outPlane * upstreamNumPlanes
                                    + upstreamPlane ) * filterSize
                                    + filterRow ) * filterSize
                                    + filterCol;
                float thisWeight = weights[thisWeightIndex];
                float thisWeightTimesError = thisWeight * thisError;
                sumWeightTimesOutError += thisWeightTimesError;
            }
        }
    }
    errorsForUpstream[globalId] = sumWeightTimesOutError;
}

// how about we make each workgroup handle one upstream plane, and iterate over examples?
// for now we assume that a workgroup is large enough to have one thread per location
// but we could always simply make each thread handle two pixels I suppose :-)
// so, workgroupId is [upstreamPlane]
// localId is [upstreamRow][upstreamCol]
// we iterate over [n]
#ifdef gOutBoardSize // for previous tests that dont define it
/*
void kernel calcErrorsForUpstream2( 
        const int batchSize,
        global const float *weightsGlobal, global const float *errorsGlobal, 
        global float *errorsForUpstreamGlobal,
        local float *_weightBoard, local float *_errorBoard ) {
    const int globalId = get_global_id(0);
    const int workgroupId = get_group_id(0);
    const int localId = get_local_id(0);
    const int workgroupSize = get_local_size(0);

    const int upstreamPlane = workgroupId;
    const int upstreamRow = localId / gUpstreamBoardSize;
    const int upstreamCol = localId % gUpstreamBoardSize;

    const int 
    if( localId < filterSizeSquared ) {
        _weightBoard[localId] = weightsGlobal[localId];
    }

    for( int n = 0; n < batchSize; n++ ) {
        float sumWeightTimesOutError = 0;
        // aggregate over [outPlane][outRow][outCol]
        for( int outPlane = 0; outPlane < outNumPlanes; outPlane++ ) {
            for( int outRow = 0; outRow < outBoardSize; outRow++ ) {
                // need to derive filterRow and filterCol, given outRow and outCol
                int filterRow = upstreamRow + margin - outRow;
                for( int outCol = 0; outCol < outBoardSize; outCol++ ) {
                   // need to derive filterRow and filterCol, given outRow and outCol
                    int filterCol = upstreamCol + margin - outCol;
                    int resultIndex = ( ( n * outNumPlanes 
                              + outPlane ) * outBoardSize
                              + outRow ) * outBoardSize
                              + outCol;
                    float thisError = errors[resultIndex];
                    int thisWeightIndex = ( ( outPlane * upstreamNumPlanes
                                        + upstreamPlane ) * filterSize
                                        + filterRow ) * filterSize
                                        + filterCol;
                    float thisWeight = weights[thisWeightIndex];
                    float thisWeightTimesError = thisWeight * thisError;
                    sumWeightTimesOutError += thisWeightTimesError;
                }
            }
        }
        errorsForUpstream[globalId] = sumWeightTimesOutError;
    }
}
*/
#endif

// so, we're just going to convolve the errorcubes with our filter cubes...
// like propagate, but easier, since no activation function, and no biases
// errorcubes (*) filters => errors
// for propagation we had:
//   images are organized like [imageId][plane][row][col]
//   filters are organized like [filterid][inplane][filterrow][filtercol]
//   results are organized like [imageid][filterid][row][col]
//   global id is organized like results, ie: [imageid][filterid][row][col]
//   - no local memory used currently
//   - each thread:
//     - loads a whole board
//     - loads a whole filter
//     - writes one output
// we will have the other way around:
//   errorcubes are organized like [imageid][outPlane][outRow][outCol]
//   filters are organized like [filterid][inplane][filterrow][filtercol]
//        (so we will swap filterid and inplane around when referencing filters, kindof)
//  globalid will be organized like upstreamresults, ie [imageid][upstreamplane][upstreamrow][upstreamcol]
#ifdef gOutBoardSize // for previous tests that dont define it
void kernel convolve_errorcubes_float( 
       const int batchSize,
      global const float *errorcubes, global const float *filters, 
    global float *upstreamErrors ) {
    int globalId = get_global_id(0);

    int upstreamBoard2Id = globalId / gUpstreamBoardSizeSquared;
    int exampleId = upstreamBoard2Id / gUpstreamNumPlanes;
    int filterId = upstreamBoard2Id % gUpstreamNumPlanes;

    if( exampleId >= batchSize ) {
        return;
    }
/*
    int errorCubeOffset = exampleId * gNumOutPlanes * gOutBoardSizeSquared;
    int filterCubeOffset = filterId * gNumInputPlanes * gFilterSizeSquared;

    int localid = globalId % upstreamBoardSizeSquared;
    int upstreamRow = localid / gUpstreamBoardSize;
    int upstreamCol = localid % gUpstreamBoardSize;

    float sum = 0;
// ====in progress
    int minm = padZeros ? max( -halfFilterSize, -outputRow ) : -halfFilterSize;
// ====to do
    int maxm = padZeros ? min( halfFilterSize, outputBoardSize - 1 - outputRow ) : halfFilterSize;
    int minn = padZeros ? max( -halfFilterSize, -outputCol ) : - halfFilterSize;
    int maxn = padZeros ? min( halfFilterSize, outputBoardSize - 1 - outputCol ) : halfFilterSize;
    int inputPlane = 0;
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
                n++;
            }
            m++;
        }
        inputPlane++;
    }
    results[globalId] = sum;*/
}
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

kernel void add_in_place( const int N, global const float*in, global float*target ) {
    int globalId = get_global_id(0);
    if( globalId < N ) {
        target[globalId] += in[globalId];
    }
}


