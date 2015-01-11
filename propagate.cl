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
#elif defined RELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
#elif defined LINEAR
    #define ACTIVATION_FUNCTION(output) (output)
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

#ifdef gOutBoardSize // for previous tests that dont define it
#ifdef ACTIVATION_FUNCTION // protect against not defined
// workgroupid [n][outputplane]
// localid: [filterrow][filtercol]
//  each thread iterates over: [inplane]
// this kernel assumes:
//   padzeros == 0 (mandatory)
//   filtersize == inputboardsize (mandatory)
//   filtersize >> outputboardsize
#if gFilterSize == gInputBoardSize && gPadZeros == 0
void kernel propagate_filter_matches_inboard( const int batchSize,
      global const float *images, global const float *filters, 
        #ifdef BIASED
            global const float*biases, 
        #endif
    global float *results,
    local float *_upstreamBoard, local float *_filterBoard ) {
    const int globalId = get_global_id(0);

    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int n = workgroupId / gNumOutPlanes;
    const int outPlane = workgroupId % gNumOutPlanes;

    const int localId = get_local_id(0);
    const int filterRow = localId / gFilterSize;
    const int filterCol = localId % gFilterSize;

//    const int minu = gPadZeros ? max( -gHalfFilterSize, -outputRow ) : -gHalfFilterSize;
//    const int maxu = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutBoardSize - 1 - outputRow  - evenPadding) : gHalfFilterSize - evenPadding;
//    const int minv = gPadZeros ? max( -gHalfFilterSize, -outputCol ) : - gHalfFilterSize;
//    const int maxv = gPadZeros ? min( gHalfFilterSize - evenPadding, gOutBoardSize - 1 - outputCol - evenPadding) : gHalfFilterSize - evenPadding;

//    const int numUpstreamsPerThread = ( gUpstreamBoardSizeSquared + workgroupSize - 1 ) / workgroupSize;
//    const int numFilterPixelsPerThread = ( gFilterSizeSquared + workgroupSize - 1 ) / workgroupSize;

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
#endif

