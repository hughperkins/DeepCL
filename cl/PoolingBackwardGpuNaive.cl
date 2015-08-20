// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// inplane and outplane are always identical, 1:1 mapping, so can just write `plane`
// gradOutput: [n][plane][outrow][outcol]
// selectors: [n][plane][outrow][outcol]
// gradInput: [n][plane][inrow][incol]
// wont use workgroups (since 'naive')
// one thread per: [n][plane][outrow][outcol]
// globalId: [n][plane][outrow][outcol]
kernel void backward(const int batchSize, 
    global const float *gradOutput, global const int *selectors, global float *gradInput) {

    #define globalId get_global_id(0)
    #define nPlaneCombo (globalId / gOutputSizeSquared) 
    #define outputPosCombo (globalId % gOutputSizeSquared)

    const int n = nPlaneCombo / gNumPlanes;
    const int plane = nPlaneCombo % gNumPlanes;
    const int outputRow = outputPosCombo / gOutputSize;
    const int outputCol = outputPosCombo % gOutputSize;

    if (n >= batchSize) {
        return;
    }

    int resultIndex = (( n
        * gNumPlanes + plane)
        * gOutputSize + outputRow)
        * gOutputSize + outputCol;
    #define error (gradOutput[resultIndex])
    int selector = (selectors[resultIndex]);
    #define drow (selector / gPoolingSize)
    #define dcol (selector % gPoolingSize)
    #define inputRow (outputRow * gPoolingSize + drow)
    #define inputCol (outputCol * gPoolingSize + dcol)
    int inputIndex = (( n
        * gNumPlanes + plane)
        * gInputSize + inputRow)
        * gInputSize + inputCol;
//    if (n < batchSize) {
        gradInput[ inputIndex ] = error;
//    }
}

