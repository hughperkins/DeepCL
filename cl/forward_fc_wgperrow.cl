// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

void copyLocal(local float *restrict target, global float const *restrict source, int N) {
    int numLoops = (N + get_local_size(0) - 1) / get_local_size(0);
    for (int loop = 0; loop < numLoops; loop++) {
        int offset = loop * get_local_size(0) + get_local_id(0);
        if (offset < N) {
            target[offset] = source[offset];
        }
    }
}

// concept:
//  we want to share each input example across multiple filters
//   but an entire filter plane is 19*19*4 = 1.4KB
//   so eg 500 filter planes is 500* 1.4KB = 700KB, much larger than local storage
//   of ~43KB
//  - we could take eg 16 filters at a time, store one filter plane from each in local storage,
//  and then bring down one example plane at a time, into local storage, during iteration over n
//  - here though, we are going to store one row from one plane from each filter, 
//  and process against one row, from same plane, from each example
//  so each workgroup will have one thread per filterId, eg 351 threads
//    each thread will add up over its assigned row
//  then, later we need to reduce over the rows
//   ... and also over the input planes?
//
// workgroupid [inputplane][filterrow]
// localid: [filterId]
//  each thread iterates over: [n][filtercol]
//  each thread is assigned to: one row, of one filter
//  workgroup is assigned to: same row, from each input plane
// local memory: one row from each output, = 128 * 19 * 4 = 9.8KB
//             1 * input row = "0.076KB"
// output1 structured as: [n][inputplane][filter][row], need to reduce again after
// this kernel assumes:
//   padzeros == 0 (mandatory)
//   filtersize == inputimagesize (mandatory)
//   inputimagesize == 19
//   filtersize == 19
//   outputSize == 1
//   lots of outplanes/filters, hundreds, but less than max work groupsize, eg 350, 500, 361
//   lots of inplanes, eg 32-128
//   inputimagesize around 19, not too small
#if (gFilterSize == gInputSize) && (gPadZeros == 0)
void kernel forward_fc_workgroup_perrow(const int batchSize,
    global const float *images, global const float *filters, 
    global float *output1,
    local float *_imageRow, local float *_filterRows) {
    const int globalId = get_global_id(0);

    const int workgroupId = get_group_id(0);
    const int workgroupSize = get_local_size(0);
    const int localId = get_local_id(0);

    const int inputPlaneId = workgroupId / gFilterSize;
    const int filterRowId = workgroupId % gFilterSize;

    const int filterId = localId;

    // first copy down filter row, which is per-thread, so we have to copy it all ourselves...
    global const float *filterRow = filters 
        + filterId * gNumInputPlanes * gFilterSizeSquared
        + inputPlaneId * gFilterSizeSquared
        + filterRowId * gFilterSize;
    local float *_threadFilterRow = _filterRows + localId * gFilterSize;
    if (localId < gNumFilters) {
        for (int i = 0; i < gFilterSize; i++) {
            _threadFilterRow[i] = filterRow[i];
        }
    }
    const int loopsPerExample = (gInputSize + workgroupSize - 1) / workgroupSize;
    // now loop over examples...
    for (int n = 0; n < batchSize; n++) {
        // copy down example row, which is global to all threads in workgroup
        // hopefully should be enough threads....
        // but we should check anyway really, since depends on number of filters configured,
        // not on relative size of filter and input image
        barrier(CLK_LOCAL_MEM_FENCE);
        copyLocal(_imageRow,  images 
            + (( n 
                * gNumInputPlanes + inputPlaneId) 
                * gInputSize + filterRowId)
                * gInputSize, 
            gInputSize);
        barrier(CLK_LOCAL_MEM_FENCE);
        // add up the values in our row...
        // note: dont activate yet, since need to reduce again
        // output structured as: [n][filter][inputplane][filterrow], need to reduce again after
        if (localId < gNumFilters) {
            float sum = 0;
            for (int filterCol = 0; filterCol < gFilterSize; filterCol++) {
                sum += _imageRow[ filterCol ] * _threadFilterRow[ filterCol ];
            }
            output1[ n * gNumInputPlanes * gNumFilters * gFilterSize
                + inputPlaneId * gFilterSize
                + filterId * gNumInputPlanes * gFilterSize + filterRowId ] = sum;
        }
    }
}
#endif

