// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR | SIGMOID | SCALEDTANH ]
// BIASED (or not)

#ifdef TANH
    #define ACTIVATION_FUNCTION(output) (tanh(output))
#elif defined SCALEDTANH
    #define ACTIVATION_FUNCTION(output) (1.7159f * tanh(0.66667f * output))
#elif SIGMOID
    #define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))
#elif defined RELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
#elif defined LINEAR
    #define ACTIVATION_FUNCTION(output) (output)
#endif

// concept:
// - assign one workgroup to each filter plane (19*19*4=1.4KB)
//   - one thread per pixel, so 361 threads
//       each filter has its own cube of filter planes
//       which cubes are by inputplane
// - loop over one input plane from each example, within each workgroup (19*19*4=1.4KB)
// - output is then non-activated sum, per input plane, per example
// number workgroups is then [num input plane (eg 64)] * [num filters (eg 361)]
// eg = 22464

// in a second kernel:
// - sum up the non-activated sum across each input plane, per example, per filter/output-plane
// - activate this
//
// workgroupid as [filterId][inputPlane]
// localid as [filterRow][filterCol]
// output as [n][filterId][inputPlane]
#if gFilterSize == gInputImagesize && gPadZeros == 0
kernel void kernel1(const int batchSize, 
    global float const * images,
    global float const * filters,
    global float *output1,
    local float *_imagePlane,
    local float *_filterPlane
) {
    const int workgroupId = get_group_id(0);
    const int localId = get_local_id(0);

    if (localId >= gFilterSizeSquared) {
        return;
    }

    const int filterId = workgroupId % gNumInputPlanes;
    const int inputPlane = workgroupId / gNumInputPlanes;
    
    // first copy down our filter plane, assume we have exactly one thread per 
    // filter plane pixel
    global float *filterPlane = filters 
        + (filterId * gNumInputPlanes + inputPlane) * gFilterSizeSquared;
    _filterPlane[localId] = filterPlane[localId];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int n = 0; n < batchSize; n++) {
        // copy down the example plane
        // oh, problem with this is, no sharing of this example across multiple filters....
    }
}
#endif

kernel void reduce() {
}


