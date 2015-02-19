// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR ]
// BIASED (or not)

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
                if( exampleId < numExamples ) {
                    sum += images[ inputboardrowoffset + inputCol] * filters[ filterrowoffset + n ];
                }
                n++;
            }
            m++;
        }
        inputPlane++;
    }

    if( exampleId < numExamples ) {
    #ifdef BIASED
        sum += biases[filterId];
    #endif
        results[globalId] = ACTIVATION_FUNCTION(sum);
    }
}
#endif

