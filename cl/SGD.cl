// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

kernel void updateWeights(
        const int N,
        const float learningRate,
        const float momentum,
        global float *lastUpdate,
        global const float *gradWeights,
        global float *weights
            ) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    // first update the update
    lastUpdate[globalId] = 
        momentum * lastUpdate[globalId]
        - learningRate * gradWeights[globalId];
    // now update the weight
    weights[globalId] += lastUpdate[globalId];
    // thats it... :-)
}

