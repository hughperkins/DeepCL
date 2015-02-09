// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "PropagateFc.h"
#include "stringhelper.h"
#include "StatefulTimer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

VIRTUAL PropagateFc::~PropagateFc() {
    delete kernel;
}
VIRTUAL void PropagateFc::propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWeightsWrapper,
//    CLWrapper *resultsWrapper ) {
//    kernel->in(batchSize);
//    kernel->input( dataWrapper );
//    kernel->input( weightsWrapper);
//    if( dim.biased ) kernel->input( biasWeightsWrapper );
//    kernel->output( resultsWrapper );
////    cout << "square(dim.outputBoardSize) " << square( dim.outputBoardSize ) << endl;
//    kernel->localFloats( square( dim.inputBoardSize ) );
//    kernel->localFloats( square( dim.filterSize ) * dim.inputPlanes );

//    int workgroupsize = std::max( 32, square( dim.outputBoardSize ) ); // no point in wasting threads....
//    int numWorkgroups = dim.numFilters * batchSize;
//    int globalSize = workgroupsize * numWorkgroups;
////    cout << "propagate3 numworkgroups " << numWorkgroups << " globalsize " << globalSize << " workgroupsize " << workgroupsize << endl;
//    kernel->run_1d( globalSize, workgroupsize );
//    cl->finish();
//    StatefulTimer::timeCheck("PropagateFc::propagate after call propagate");
}
PropagateFc::PropagateFc( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn ) :
        Propagate( cl, dim, fn )
            {

    std::string options = "-D " + fn->getDefineName();
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/propagate_fc.cl", "propagate_fc", 'options' )
    // ]]]
    // generated using cog:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "// expected defines:\n" 
    "// one of: [ TANH | RELU | LINEAR ]\n" 
    "// BIASED (or not)\n" 
    "\n" 
    "#ifdef TANH\n" 
    "#define ACTIVATION_FUNCTION(output) (tanh(output))\n" 
    "#elif defined SCALEDTANH\n" 
    "#define ACTIVATION_FUNCTION(output) ( 1.7159f * tanh( 0.66667f * output))\n" 
    "#elif SIGMOID\n" 
    "#define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))\n" 
    "#elif defined RELU\n" 
    "#define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)\n" 
    "#elif defined LINEAR\n" 
    "#define ACTIVATION_FUNCTION(output) (output)\n" 
    "#endif\n" 
    "\n" 
    "#ifdef gOutBoardSize // for previous tests that dont define it\n" 
    "#ifdef ACTIVATION_FUNCTION // protect against not defined\n" 
    "// workgroupid [n]\n" 
    "// localid: [outputplane]\n" 
    "//  each thread iterates over: [inputplane]\n" 
    "// this kernel assumes:\n" 
    "//   padzeros == 0 (mandatory)\n" 
    "//   filtersize == inputboardsize (mandatory)\n" 
    "//   inputboardsize == 19\n" 
    "//   filtersize == 19\n" 
    "//   outputBoardSize == 1\n" 
    "//   lots of outplanes, hundreds, but less than max work groupsize, eg 350, 500, 361\n" 
    "//   lots of inplanes, eg 32\n" 
    "//   inputboardsize around 19, not too small\n" 
    "#if gFilterSize == gInputBoardSize && gPadZeros == 0\n" 
    "void kernel propagate_fc( const int batchSize,\n" 
    "global const float *images, global const float *filters,\n" 
    "#ifdef BIASED\n" 
    "global const float*biases,\n" 
    "#endif\n" 
    "global float *results,\n" 
    "local float *_upstreamBoard, local float *_filterBoard ) {\n" 
    "const int globalId = get_global_id(0);\n" 
    "\n" 
    "const int workgroupId = get_group_id(0);\n" 
    "const int workgroupSize = get_local_size(0);\n" 
    "const int n = workgroupId / gNumOutPlanes;\n" 
    "const int outPlane = workgroupId % gNumOutPlanes;\n" 
    "\n" 
    "const int localId = get_local_id(0);\n" 
    "const int filterRow = localId / gFilterSize;\n" 
    "const int filterCol = localId % gFilterSize;\n" 
    "\n" 
    "float sum = 0;\n" 
    "for( int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++ ) {\n" 
    "int thisUpstreamBoardOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;\n" 
    "barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "for( int i = 0; i < numUpstreamsPerThread; i++ ) {\n" 
    "int thisOffset = workgroupSize * i + localId;\n" 
    "if( thisOffset < gUpstreamBoardSizeSquared ) {\n" 
    "_upstreamBoard[ thisOffset ] = images[ thisUpstreamBoardOffset + thisOffset ];\n" 
    "}\n" 
    "}\n" 
    "const int filterGlobalOffset = ( outPlane * gUpstreamNumPlanes + upstreamPlane ) * gFilterSizeSquared;\n" 
    "for( int i = 0; i < numFilterPixelsPerThread; i++ ) {\n" 
    "int thisOffset = workgroupSize * i + localId;\n" 
    "if( thisOffset < gFilterSizeSquared ) {\n" 
    "_filterCube[thisOffset] = filters[filterGlobalOffset + thisOffset];\n" 
    "}\n" 
    "}\n" 
    "barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "if( localId < gOutBoardSizeSquared ) {\n" 
    "for( int u = minu; u <= maxu; u++ ) {\n" 
    "int inputRow = outputRow + u + ( gPadZeros ? 0 : gHalfFilterSize );\n" 
    "int inputboardrowoffset = inputRow * gUpstreamBoardSize;\n" 
    "int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n" 
    "for( int v = minv; v <= maxv; v++ ) {\n" 
    "int inputCol = outputCol + v + ( gPadZeros ? 0 : gHalfFilterSize );\n" 
    "sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];\n" 
    "}\n" 
    "}\n" 
    "}\n" 
    "}\n" 
    "#ifdef BIASED\n" 
    "sum += biases[outPlane];\n" 
    "#endif\n" 
    "// results are organized like [imageid][filterid][row][col]\n" 
    "int resultIndex = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared + localId;\n" 
    "if( localId < gOutBoardSizeSquared ) {\n" 
    "results[resultIndex ] = ACTIVATION_FUNCTION(sum);\n" 
    "//        results[resultIndex ] = 123;\n" 
    "}\n" 
    "}\n" 
    "#endif\n" 
    "#endif\n" 
    "#endif\n" 
    "\n" 
    "#ifdef gOutBoardSize // for previous tests that dont define it\n" 
    "#ifdef ACTIVATION_FUNCTION // protect against not defined\n" 
    "// workgroupid [n][outputplane]\n" 
    "// localid: [filterrow][filtercol]\n" 
    "//  each thread iterates over: [inplane]\n" 
    "// this kernel assumes:\n" 
    "//   padzeros == 0 (mandatory)\n" 
    "//   filtersize == inputboardsize (mandatory)\n" 
    "//   outputBoardSize == 1\n" 
    "//   lots of outplanes, hundreds, but less than max work groupsize, eg 350, 500, 361\n" 
    "//   lots of inplanes, eg 32\n" 
    "//   inputboardsize around 19, not too small\n" 
    "#if gFilterSize == gInputBoardSize && gPadZeros == 0\n" 
    "void kernel propagate_filter_matches_inboard( const int batchSize,\n" 
    "global const float *images, global const float *filters,\n" 
    "#ifdef BIASED\n" 
    "global const float*biases,\n" 
    "#endif\n" 
    "global float *results,\n" 
    "local float *_upstreamBoard, local float *_filterBoard ) {\n" 
    "const int globalId = get_global_id(0);\n" 
    "\n" 
    "const int workgroupId = get_group_id(0);\n" 
    "const int workgroupSize = get_local_size(0);\n" 
    "const int n = workgroupId / gNumOutPlanes;\n" 
    "const int outPlane = workgroupId % gNumOutPlanes;\n" 
    "\n" 
    "const int localId = get_local_id(0);\n" 
    "const int filterRow = localId / gFilterSize;\n" 
    "const int filterCol = localId % gFilterSize;\n" 
    "\n" 
    "float sum = 0;\n" 
    "for( int upstreamPlane = 0; upstreamPlane < gUpstreamNumPlanes; upstreamPlane++ ) {\n" 
    "int thisUpstreamBoardOffset = ( n * gUpstreamNumPlanes + upstreamPlane ) * gUpstreamBoardSizeSquared;\n" 
    "barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "for( int i = 0; i < numUpstreamsPerThread; i++ ) {\n" 
    "int thisOffset = workgroupSize * i + localId;\n" 
    "if( thisOffset < gUpstreamBoardSizeSquared ) {\n" 
    "_upstreamBoard[ thisOffset ] = images[ thisUpstreamBoardOffset + thisOffset ];\n" 
    "}\n" 
    "}\n" 
    "const int filterGlobalOffset = ( outPlane * gUpstreamNumPlanes + upstreamPlane ) * gFilterSizeSquared;\n" 
    "for( int i = 0; i < numFilterPixelsPerThread; i++ ) {\n" 
    "int thisOffset = workgroupSize * i + localId;\n" 
    "if( thisOffset < gFilterSizeSquared ) {\n" 
    "_filterCube[thisOffset] = filters[filterGlobalOffset + thisOffset];\n" 
    "}\n" 
    "}\n" 
    "barrier(CLK_LOCAL_MEM_FENCE);\n" 
    "if( localId < gOutBoardSizeSquared ) {\n" 
    "for( int u = minu; u <= maxu; u++ ) {\n" 
    "int inputRow = outputRow + u + ( gPadZeros ? 0 : gHalfFilterSize );\n" 
    "int inputboardrowoffset = inputRow * gUpstreamBoardSize;\n" 
    "int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n" 
    "for( int v = minv; v <= maxv; v++ ) {\n" 
    "int inputCol = outputCol + v + ( gPadZeros ? 0 : gHalfFilterSize );\n" 
    "sum += _upstreamBoard[ inputboardrowoffset + inputCol] * _filterCube[ filterrowoffset + v ];\n" 
    "}\n" 
    "}\n" 
    "}\n" 
    "}\n" 
    "#ifdef BIASED\n" 
    "sum += biases[outPlane];\n" 
    "#endif\n" 
    "// results are organized like [imageid][filterid][row][col]\n" 
    "int resultIndex = ( n * gNumOutPlanes + outPlane ) * gOutBoardSizeSquared + localId;\n" 
    "if( localId < gOutBoardSizeSquared ) {\n" 
    "results[resultIndex ] = ACTIVATION_FUNCTION(sum);\n" 
    "//        results[resultIndex ] = 123;\n" 
    "}\n" 
    "}\n" 
    "#endif\n" 
    "#endif\n" 
    "#endif\n" 
    "\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "propagate_fc", options, "cl/propagate_fc.cl" );
    // [[[end]]]
}

