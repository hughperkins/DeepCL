// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "ForwardFc.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL ForwardFc::~ForwardFc() {
    delete kernel1;
    delete kernel_reduce;
//    delete kernel_activate;
    delete kPerElementTiledAdd;
}
VIRTUAL void ForwardFc::forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper, CLWrapper *outputWrapper ) {
    StatefulTimer::timeCheck("ForwardFc::forward begin");

    const int maxWorkgroupSize = cl->getMaxWorkgroupSize();

    const int output1Size = batchSize * dim.numFilters * dim.numInputPlanes * dim.filterSize;
    float *output1 = new float[ output1Size ];
    CLWrapper *output1Wrapper = cl->wrap( output1Size, output1 );
    output1Wrapper->createOnDevice();

    const int output2Size = batchSize * dim.numFilters * dim.numInputPlanes;
    float *output2 = new float[ output2Size ];
    CLWrapper *output2Wrapper = cl->wrap( output2Size, output2 );
    output2Wrapper->createOnDevice();

    kernel1->in(batchSize);
    kernel1->input( dataWrapper );
    kernel1->input( weightsWrapper);
//    if( dim.biased ) kernel1->input( biasWrapper );
    kernel1->output( output1Wrapper );
    kernel1->localFloats( dim.inputImageSize );
    kernel1->localFloats( dim.numFilters * dim.filterSize );

    int workgroupSize = dim.numFilters;
    int numWorkgroups = dim.filterSize * dim.numInputPlanes;

    int globalSize = workgroupSize * numWorkgroups;
    kernel1->run_1d( globalSize, workgroupSize );
    cl->finish();
    StatefulTimer::timeCheck("ForwardFc::forward after first kernel");

    // now reduce over rows 
    kernel_reduce->in(batchSize * dim.numFilters * dim.numInputPlanes)
        ->in( dim.filterSize )
        ->in( output1Wrapper )->out( output2Wrapper );
    int maxglobalId = batchSize * dim.numFilters * dim.numInputPlanes;
//    numWorkgroups = ( maxglobalId + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
//    kernel_reduce->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
    numWorkgroups = ( maxglobalId + 64 - 1 ) / 64;
    kernel_reduce->run_1d( numWorkgroups * 64, 64 );
    cl->finish();
    StatefulTimer::timeCheck("ForwardFc::forward after reduce1");

    // reduce over input planes 
    kernel_reduce->in(batchSize * dim.numFilters)->in( dim.numInputPlanes )
        ->in( output2Wrapper )->out( outputWrapper );
    maxglobalId = batchSize * dim.numFilters;
    numWorkgroups = ( batchSize * dim.numFilters + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
    kernel_reduce->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
//    numWorkgroups = ( maxglobalId + 64 - 1 ) / 64;
//    kernel_reduce->run_1d( numWorkgroups * 64, 64 );
    cl->finish();
    StatefulTimer::timeCheck("ForwardFc::forward after reduce2");

    // add bias...
    if( dim.biased ) {
        kPerElementTiledAdd->in( batchSize * dim.numFilters )->in( dim.numFilters )->inout( outputWrapper )->in( biasWrapper );
        maxglobalId = batchSize * dim.numFilters;
        numWorkgroups = ( batchSize * dim.numFilters + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
        kPerElementTiledAdd->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
        cl->finish();
        StatefulTimer::timeCheck("ForwardFc::forward after add bias");        
    }

//    kernel_activate->in( batchSize * dim.numFilters )
//        ->inout( outputWrapper );
//    maxglobalId = batchSize * dim.numFilters;
//    numWorkgroups = ( batchSize * dim.numFilters + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
//    kernel_activate->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
//    cl->finish();
//    StatefulTimer::timeCheck("ForwardFc::forward after activate");

    delete output2Wrapper;
    delete[] output2;

    delete output1Wrapper;
    delete[] output1;
    StatefulTimer::timeCheck("ForwardFc::forward end");
}
ForwardFc::ForwardFc( EasyCL *cl, LayerDimensions dim ) :
        Forward( cl, dim )
            {

    if( dim.inputImageSize != dim.filterSize ) {
        throw runtime_error("For ForwardFc, filtersize and inputimagesize must be identical");
    }
    if( dim.padZeros ) {
        throw runtime_error("For ForwardFc, padzeros must be disabled");
    }

    std::string options = ""; // "-D " + fn->getDefineName();
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel1", "cl/forward_fc_wgperrow.cl", "forward_fc_workgroup_perrow", 'options' )
    // stringify.write_kernel2( "kernel_reduce", "cl/reduce_segments.cl", "reduce_segments", 'options' )
    // # stringify.write_kernel2( "kernel_activate", "cl/activate.cl", "activate", 'options' )
    // # stringify.write_kernel2( "kPerElementAdd", "cl/per_element_add.cl", "per_element_add", 'options' )
    // stringify.write_kernel2( "kPerElementTiledAdd", "cl/per_element_add.cl", "per_element_tiled_add", 'options' )
    // ]]]
    // generated using cog, from cl/forward_fc_wgperrow.cl:
    const char * kernel1Source =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "void copyLocal( local float *restrict target, global float const *restrict source, int N ) {\n" 
    "    int numLoops = ( N + get_local_size(0) - 1 ) / get_local_size(0);\n" 
    "    for( int loop = 0; loop < numLoops; loop++ ) {\n" 
    "        int offset = loop * get_local_size(0) + get_local_id(0);\n" 
    "        if( offset < N ) {\n" 
    "            target[offset] = source[offset];\n" 
    "        }\n" 
    "    }\n" 
    "}\n" 
    "\n" 
    "// concept:\n" 
    "//  we want to share each input example across multiple filters\n" 
    "//   but an entire filter plane is 19*19*4 = 1.4KB\n" 
    "//   so eg 500 filter planes is 500* 1.4KB = 700KB, much larger than local storage\n" 
    "//   of ~43KB\n" 
    "//  - we could take eg 16 filters at a time, store one filter plane from each in local storage,\n" 
    "//  and then bring down one example plane at a time, into local storage, during iteration over n\n" 
    "//  - here though, we are going to store one row from one plane from each filter,\n" 
    "//  and process against one row, from same plane, from each example\n" 
    "//  so each workgroup will have one thread per filterId, eg 351 threads\n" 
    "//    each thread will add up over its assigned row\n" 
    "//  then, later we need to reduce over the rows\n" 
    "//   ... and also over the input planes?\n" 
    "//\n" 
    "// workgroupid [inputplane][filterrow]\n" 
    "// localid: [filterId]\n" 
    "//  each thread iterates over: [n][filtercol]\n" 
    "//  each thread is assigned to: one row, of one filter\n" 
    "//  workgroup is assigned to: same row, from each input plane\n" 
    "// local memory: one row from each output, = 128 * 19 * 4 = 9.8KB\n" 
    "//             1 * input row = \"0.076KB\"\n" 
    "// output1 structured as: [n][inputplane][filter][row], need to reduce again after\n" 
    "// this kernel assumes:\n" 
    "//   padzeros == 0 (mandatory)\n" 
    "//   filtersize == inputimagesize (mandatory)\n" 
    "//   inputimagesize == 19\n" 
    "//   filtersize == 19\n" 
    "//   outputImageSize == 1\n" 
    "//   lots of outplanes/filters, hundreds, but less than max work groupsize, eg 350, 500, 361\n" 
    "//   lots of inplanes, eg 32-128\n" 
    "//   inputimagesize around 19, not too small\n" 
    "#if (gFilterSize == gInputImageSize) && (gPadZeros == 0)\n" 
    "void kernel forward_fc_workgroup_perrow( const int batchSize,\n" 
    "    global const float *images, global const float *filters,\n" 
    "    global float *output1,\n" 
    "    local float *_imageRow, local float *_filterRows ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "\n" 
    "    const int workgroupId = get_group_id(0);\n" 
    "    const int workgroupSize = get_local_size(0);\n" 
    "    const int localId = get_local_id(0);\n" 
    "\n" 
    "    const int inputPlaneId = workgroupId / gFilterSize;\n" 
    "    const int filterRowId = workgroupId % gFilterSize;\n" 
    "\n" 
    "    const int filterId = localId;\n" 
    "\n" 
    "    // first copy down filter row, which is per-thread, so we have to copy it all ourselves...\n" 
    "    global const float *filterRow = filters\n" 
    "        + filterId * gNumInputPlanes * gFilterSizeSquared\n" 
    "        + inputPlaneId * gFilterSizeSquared\n" 
    "        + filterRowId * gFilterSize;\n" 
    "    local float *_threadFilterRow = _filterRows + localId * gFilterSize;\n" 
    "    for( int i = 0; i < gFilterSize; i++ ) {\n" 
    "        _threadFilterRow[i] = filterRow[i];\n" 
    "    }\n" 
    "    const int loopsPerExample = ( gInputImageSize + workgroupSize - 1 ) / workgroupSize;\n" 
    "    // now loop over examples...\n" 
    "    for( int n = 0; n < batchSize; n++ ) {\n" 
    "        // copy down example row, which is global to all threads in workgroup\n" 
    "        // hopefully should be enough threads....\n" 
    "        // but we should check anyway really, since depends on number of filters configured,\n" 
    "        // not on relative size of filter and input image\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        copyLocal( _imageRow,  images\n" 
    "            + ( ( n\n" 
    "                * gNumInputPlanes + inputPlaneId )\n" 
    "                * gInputImageSize + filterRowId )\n" 
    "                * gInputImageSize,\n" 
    "            gInputImageSize );\n" 
    "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "        // add up the values in our row...\n" 
    "        float sum = 0;\n" 
    "        for( int filterCol = 0; filterCol < gFilterSize; filterCol++ ) {\n" 
    "            sum += _imageRow[ filterCol ] * _threadFilterRow[ filterCol ];\n" 
    "        }\n" 
    "        // note: dont activate yet, since need to reduce again\n" 
    "        // output structured as: [n][filter][inputplane][filterrow], need to reduce again after\n" 
    "        if( localId < gNumFilters ) {\n" 
    "            output1[ n * gNumInputPlanes * gNumFilters * gFilterSize\n" 
    "                + inputPlaneId * gFilterSize\n" 
    "                + filterId * gNumInputPlanes * gFilterSize + filterRowId ] = sum;\n" 
    "        }\n" 
    "    }\n" 
    "}\n" 
    "#endif\n" 
    "\n" 
    "";
    kernel1 = cl->buildKernelFromString( kernel1Source, "forward_fc_workgroup_perrow", options, "cl/forward_fc_wgperrow.cl" );
    // generated using cog, from cl/reduce_segments.cl:
    const char * kernel_reduceSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void reduce_segments( const int numSegments, const int segmentLength,\n" 
    "        global float const *in, global float* out ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    const int segmentId = globalId;\n" 
    "\n" 
    "    if( segmentId >= numSegments ) {\n" 
    "        return;\n" 
    "    }\n" 
    "\n" 
    "    float sum = 0;\n" 
    "    global const float *segment = in + segmentId * segmentLength;\n" 
    "    for( int i = 0; i < segmentLength; i++ ) {\n" 
    "        sum += segment[i];\n" 
    "    }\n" 
    "    out[segmentId] = sum;\n" 
    "}\n" 
    "\n" 
    "\n" 
    "";
    kernel_reduce = cl->buildKernelFromString( kernel_reduceSource, "reduce_segments", options, "cl/reduce_segments.cl" );
    // generated using cog, from cl/per_element_add.cl:
    const char * kPerElementTiledAddSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void per_element_add( const int N, global float *target, global const float *source ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    target[globalId] += source[globalId];\n" 
    "}\n" 
    "\n" 
    "// adds source to target\n" 
    "// tiles source as necessary, according to tilingSize\n" 
    "kernel void per_element_tiled_add( const int N, const int tilingSize, global float *target, global const float *source ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    target[globalId] += source[globalId % tilingSize];\n" 
    "}\n" 
    "\n" 
    "kernel void repeated_add( const int N, const int sourceSize, const int repeatSize, global float *target, global const float *source ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    target[globalId] += source[ ( globalId / repeatSize ) % sourceSize ];\n" 
    "}\n" 
    "\n" 
    "";
    kPerElementTiledAdd = cl->buildKernelFromString( kPerElementTiledAddSource, "per_element_tiled_add", options, "cl/per_element_add.cl" );
    // [[[end]]]
}

