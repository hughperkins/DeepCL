// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// notes on non-odd filtersizes:
// for odd, imagesize and filtersize 3, padZeros = 0:
// output is a single square
// m and n should vary between -1,0,1
// for even, imagesize and filtersize 2, padzeros = 0
// output is a single square, which we can position at topleft or bottomrigth
// lets position it in bottomright
// then m and n should vary as -1,0
//
// for even, imagesize and filtersize 2, padzeros = 1
// output is 2 by 2
// well... if it is even:
// - if we are not padding zeros, then we simply move our filter around the image somehow
// - if we are padding zeros, then we conceptually pad the bottom and right edge of the image with zeros by 1
// filtersize remains the same
//      m will vary as -1,0,1
//       outputrow is fixed by globalid
//       inputrow should be unchanged...
// padzeros = 0:
//  x x .  . . .
//  x x .  . x x
//  . . .  . x x
// when filtersize even:
//    new imagesize = oldimagesize - filtersize + 1
// when filtersize odd:
//    x x x .
//    x x x .
//    x x x .
//    . . . .
//    new imagesize = oldimagesize - filtersize + 1
// padzeros = 1:
// x x
// x x . .   x x .    . . .     . . .
//   . . .   x x .    . x x     . . .
//   . . .   . . .    . x x     . . x x
// outrow=0 outrow=1  outrow=2      x x
// outcol=0 outcol=1  outcol=2    outrow=3
//                                outcol=3
// when filtersize is even, and padzeros, imagesize grows by 1 each time...
//    imagesize = oldimagesize + 1
// when filtersize is odd
//  x x x 
//  x x x .   x x x    . . .
//  x x x .   x x x    . x x x
//    . . .   x x x    . x x x
//                       x x x

// images are organized like [imageId][plane][row][col]
// filters are organized like [filterid][inplane][filterrow][filtercol]
// output are organized like [imageid][filterid][row][col]
// global id is organized like output, ie: [imageid][outplane][outrow][outcol]
// - no local memory used currently
// - each thread:
//     - loads a whole upstream cube
//     - loads a whole filter cube
//     - writes one output...
void kernel convolve_imagecubes_float2(
    const int numExamples,
      global const float *inputs, global const float *filters, 
    global float *output) {
    int globalId = get_global_id(0);

    int outputImage2Id = globalId / gOutputSizeSquared;
    int exampleId = outputImage2Id / gNumFilters;
    int filterId = outputImage2Id % gNumFilters;

    // intraimage coords
    int localid = globalId % gOutputSizeSquared;
    int outputRow = localid / gOutputSize;
    int outputCol = localid % gOutputSize;

    global float const*inputCube = inputs + exampleId * gNumInputPlanes * gInputSizeSquared;
    global float const*filterCube = filters + filterId * gNumInputPlanes * gFilterSizeSquared;

    float sum = 0;
    if (exampleId < numExamples) {
        for (int inputPlaneIdx = 0; inputPlaneIdx < gNumInputPlanes; inputPlaneIdx++) {
            global float const*inputPlane = inputCube + inputPlaneIdx * gInputSizeSquared;
            global float const*filterPlane = filterCube + inputPlaneIdx * gFilterSizeSquared;
            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {
                // trying to reduce register pressure...
                #if gPadZeros == 1
                    #define inputRowIdx (outputRow + u)
                #else
                    #define inputRowIdx (outputRow + u + gHalfFilterSize)
                #endif
                global float const *inputRow = inputPlane + inputRowIdx * gInputSize;
                global float const *filterRow = filterPlane + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                bool rowOk = inputRowIdx >= 0 && inputRowIdx < gInputSize;
                #pragma unroll
                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {
                    #if gPadZeros == 1
                        #define inputColIdx (outputCol + v)
                    #else
                        #define inputColIdx (outputCol + v + gHalfFilterSize)
                    #endif
                    bool process = rowOk && inputColIdx >= 0 && inputColIdx < gInputSize;
                    if (process) {
                            sum += inputRow[inputColIdx] * filterRow[v];
                    }
                }
            }
        }
    }

    if (exampleId < numExamples) {
        output[globalId] = sum;
    }
}

