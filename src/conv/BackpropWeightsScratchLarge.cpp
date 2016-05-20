// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "BackpropWeightsScratchLarge.h"
#include "util/StatefulTimer.h"
#include "util/stringhelper.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

VIRTUAL BackpropWeightsScratchLarge::~BackpropWeightsScratchLarge() {
    delete kernel;
}
VIRTUAL void BackpropWeightsScratchLarge::calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper) {
    StatefulTimer::instance()->timeCheck("BackpropWeightsScratchLarge start");

    int workgroupSize = 32 * (( square(dim.filterSize) + 32 - 1) / 32); // quantize to nearest 32
//    int workgroupsize = std::max(32, square(dim.filterSize) ); // no point in wasting cores...
    int numWorkgroups = dim.inputPlanes * dim.numFilters;
    int globalSize = workgroupSize * numWorkgroups;
//    globalSize = (( globalSize + workgroupSize - 1) / workgroupSize) * workgroupSize;
//    cout << "workgroupsize " << workgroupSize << " numworkgroups " << numWorkgroups << " globalsize " << globalSize << endl;

    const float learningMultiplier = learningRateToMultiplier(batchSize);

    kernel
       ->in(learningMultiplier)
       ->in(batchSize)
       ->in(gradOutputWrapper)
        ->in(imagesWrapper)
       ->inout(gradWeightsWrapper);
    if(dim.biased) {
        kernel->inout(gradBiasWrapper);
    }
    kernel
        ->localFloats(outputStripeSize)
        ->localFloats(inputStripeOuterSize);

    kernel->run_1d(globalSize, workgroupSize);

    cl->finish();

    StatefulTimer::instance()->timeCheck("BackpropWeightsScratchLarge end");
}
BackpropWeightsScratchLarge::BackpropWeightsScratchLarge(EasyCL *cl, LayerDimensions dim) :
        BackpropWeights(cl, dim)
            {
    if(square(dim.filterSize) > cl->getMaxWorkgroupSize()) {
        throw runtime_error("cannot use BackpropWeightsScratchLarge, since filterSize * filterSize > maxworkgroupsize");
    }

    // [[[cog
    // import stringify
    // # stringify.write_kernel("kernelSource", "ClConvolve.cl")
    // ]]]
    // [[[end]]]
//    cout << "dim: " << dim << endl;
    std::string options = dim.buildOptionsString();

    int localMemoryRequirementsFullImage = dim.inputSize * dim.inputSize * 4 + dim.outputSize * dim.outputSize * 4;
    int availableLocal = cl->getLocalMemorySize();
//    cout << "localmemoryrequirementsfullimage: " << localMemoryRequirementsFullImage << endl;
//    cout << "availablelocal: " << availableLocal << endl;
    // make the local memory used about one quarter of what is available? half of what is available?
    // let's try one quarter :-)
    int localWeCanUse = availableLocal / 4;
    numStripes = (localMemoryRequirementsFullImage + localWeCanUse - 1) / localWeCanUse;
//    cout << "numStripes: " << numStripes << endl;
    // make it a power of 2
    numStripes = EasyCL::getNextPower2(numStripes);
//    cout << "numStripes: " << numStripes << endl;

    int inputStripeMarginRows = dim.filterSize - 1;
    int inputStripeInnerNumRows = dim.inputSize / numStripes;
    int inputStripeOuterNumRows = inputStripeInnerNumRows + 2 * inputStripeMarginRows;

    int inputStripeInnerSize = inputStripeInnerNumRows * dim.inputSize;
    inputStripeOuterSize = inputStripeOuterNumRows * dim.inputSize;
    int inputStripeMarginSize = inputStripeMarginRows * dim.inputSize;

    int outputStripeNumRows = (dim.outputSize + numStripes - 1) / numStripes;
    outputStripeSize = outputStripeNumRows * dim.outputSize;

    // [[[cog
    // import cog_optionswriter
    // cog_optionswriter.write_options(['numStripes','inputStripeMarginRows','inputStripeInnerNumRows',
    //     'inputStripeOuterNumRows', 'inputStripeInnerSize', 'inputStripeOuterSize', 'inputStripeMarginSize',
    //     'outputStripeNumRows', 'outputStripeSize' ])
    // ]]]
    // generated, using cog:
    options += " -DgNumStripes=" + toString(numStripes);
    options += " -DgInputStripeMarginRows=" + toString(inputStripeMarginRows);
    options += " -DgInputStripeInnerNumRows=" + toString(inputStripeInnerNumRows);
    options += " -DgInputStripeOuterNumRows=" + toString(inputStripeOuterNumRows);
    options += " -DgInputStripeInnerSize=" + toString(inputStripeInnerSize);
    options += " -DgInputStripeOuterSize=" + toString(inputStripeOuterSize);
    options += " -DgInputStripeMarginSize=" + toString(inputStripeMarginSize);
    options += " -DgOutputStripeNumRows=" + toString(outputStripeNumRows);
    options += " -DgOutputStripeSize=" + toString(outputStripeSize);
    // [[[end]]]
    cout << "options: " << options << endl;

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/fsword73_backpropweights_fast.cl", "test_kernel", 'options')
    // ]]]
    // generated using cog, from cl/fsword73_backpropweights_fast.cl:
    const char * kernelSource =  
    "// Copyright // FSWORD73 AT HOTMAIL DOT COM\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// expected defines:\n"
    "// BIASED (or not)\n"
    "\n"
    "// Original workgroup by  hughperkins at gmail in deepCL\n"
    "// globalId: [outPlane][inputPlane][filterRow][filterCol]\n"
    "// per-thread iteration: [n][outputRow][outputCol]\n"
    "//27     int workgroupsize = std::max(32, square(dim.filterSize) ); // no point in wasting cores...\n"
    "//28     int numWorkgroups = dim.inputPlanes * dim.numFilters;\n"
    "//29     int globalSize = workgroupsize * numWorkgroups;\n"
    "//30     globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;\n"
    "\n"
    "#define 	gNumFilters 	8\n"
    "#define 	gInputPlanes	4\n"
    "#define   gFilterSize   3\n"
    "#define   gFilterSizeSquared (gFilterSize*gFilterSize)\n"
    "//#define   gOutputSize   128\n"
    "#define   gInputSize    128\n"
    "#define   gMargin        0\n"
    "#define   FIXED_WORKGROUPSIZE 64\n"
    "#define   FIXED_WORKGROUPSIZE_SHIFT 6\n"
    "\n"
    "void  backprop_floats_fast_valid_thread(\n"
    "        global const float *gradOutput,\n"
    "				global const float *images,\n"
    "				float *thiswchange,\n"
    "				#ifdef BIASED\n"
    "				  float*  thisbiaschange,\n"
    "				#endif\n"
    "				const int batchSize,\n"
    "				const int gOutputSize,\n"
    "				int localId,\n"
    "				int globalId\n"
    " )\n"
    " {\n"
    "			  *thiswchange = 0;\n"
    "				int batchId     =  (globalId >> FIXED_WORKGROUPSIZE_SHIFT) % batchSize  ;\n"
    "			  int globalIdOutput = (globalId >> FIXED_WORKGROUPSIZE_SHIFT) / batchSize ;\n"
    "\n"
    "				int IntraFilterOffset =  globalIdOutput % gFilterSizeSquared;\n"
    "				int filterRow = IntraFilterOffset / gFilterSize;\n"
    "				int filterCol = IntraFilterOffset % gFilterSize;\n"
    "\n"
    "				int filter2Id = globalIdOutput / gFilterSizeSquared;\n"
    "				int outPlane = filter2Id / gInputPlanes;\n"
    "				int upstreamPlane = filter2Id % gInputPlanes;\n"
    "				int n = batchId;\n"
    "\n"
    "\n"
    "			  int iterations = (gOutputSize * gOutputSize + FIXED_WORKGROUPSIZE -1) >> FIXED_WORKGROUPSIZE_SHIFT;\n"
    "\n"
    "	      for(int i = 0; i < iterations; i++)\n"
    "				{\n"
    "						int offsetofOutput = i*FIXED_WORKGROUPSIZE + localId;\n"
    "\n"
    "						if( offsetofOutput < (gOutputSize * gOutputSize))\n"
    "						{\n"
    "								int outRow = offsetofOutput / gOutputSize;\n"
    "								int upstreamRow = outRow - gMargin + filterRow;\n"
    "								int outCol = offsetofOutput % gOutputSize;\n"
    "								int upstreamCol = outCol - gMargin + filterCol;\n"
    "								bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputSize\n"
    "										 && upstreamCol < gInputSize;\n"
    "\n"
    "								if (proceed) {\n"
    "											int resultIndex = (( n * gNumFilters\n"
    "																+ outPlane) * gOutputSize\n"
    "																+ outRow) * gOutputSize\n"
    "																+ outCol;\n"
    "											float error = gradOutput[resultIndex];\n"
    "											int upstreamDataIndex = (( n * gInputPlanes\n"
    "																			 + upstreamPlane) * gInputSize\n"
    "																			 + upstreamRow) * gInputSize\n"
    "																			 + upstreamCol;\n"
    "											float upstreamResult = images[upstreamDataIndex];\n"
    "											float thisimagethiswchange = upstreamResult * error;\n"
    "											*thiswchange += thisimagethiswchange;\n"
    "			#ifdef BIASED\n"
    "											*thisbiaschange += error;\n"
    "			#endif\n"
    "									}\n"
    "						}\n"
    "				}		//for loop\n"
    " }\n"
    "\n"
    "\n"
    " void Reduction_of_Weights(\n"
    "				float* thiswchange,\n"
    "				__local  float* sdata,\n"
    "				#ifdef BIASED\n"
    "				 float*  thisbiaschange,\n"
    "				#endif\n"
    "			  int localId,\n"
    "				int globalId)\n"
    " {\n"
    "\n"
    "			//store into local\n"
    "			sdata[localId] = *thiswchange;\n"
    "			barrier(CLK_LOCAL_MEM_FENCE);\n"
    "			for(unsigned int s = FIXED_WORKGROUPSIZE >>1; s > 0; s >>= 1)\n"
    "			{\n"
    "        if(localId < s)\n"
    "        {\n"
    "            sdata[localId] += sdata[localId + s];\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "			}\n"
    "\n"
    "			if(localId == 0)\n"
    "			{\n"
    "				*thiswchange = 	sdata[0];\n"
    "			}\n"
    "#ifdef BIASED\n"
    "			sdata[localId] = *thisbiaschange;\n"
    "			barrier(CLK_LOCAL_MEM_FENCE);\n"
    "			for(unsigned int s = FIXED_WORKGROUPSIZE >>1; s > 0; s >>= 1)\n"
    "			{\n"
    "        if(localId < s)\n"
    "        {\n"
    "            sdata[localId] += sdata[localId + s];\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "			}\n"
    "\n"
    "			if(localId == 0)\n"
    "			{\n"
    "				*thisbiaschange = 	sdata[0];\n"
    "			}\n"
    "#endif\n"
    "}\n"
    "\n"
    "\n"
    "inline void AtomicAdd(volatile __global float *source, const float operand) {\n"
    "    union {\n"
    "        unsigned int intVal;\n"
    "        float floatVal;\n"
    "    } newVal;\n"
    "    union {\n"
    "        unsigned int intVal;\n"
    "        float floatVal;\n"
    "    } prevVal;\n"
    "    do {\n"
    "        prevVal.floatVal = *source;\n"
    "        newVal.floatVal = prevVal.floatVal + operand;\n"
    "    } while (atomic_cmpxchg((volatile __global unsigned int *)source,\n"
    "                             prevVal.intVal, newVal.intVal)\n"
    "                             != prevVal.intVal);\n"
    "}\n"
    "\n"
    "//\n"
    "//   Global ID :  [outPlane][inputPlane][filterRow][filterCol] *  BatchSize * 64\n"
    "//   Per thread = ( gOutputSize * gOutputSize)/64\n"
    "//   workgroupsize =  64 treads == 1 image\n"
    "//   numWorkgroups  = [outPlane][inputPlane][filterRow][filterCol] *  BatchSize\n"
    "//   int globalSize = workgroupsize * numWorkgroups;\n"
    "//   Assuming   [outPlane][inputPlane][filterRow][filterCol] is cleared or preset\n"
    "//   Each workgroupsize call Atomic_Add\n"
    "//	  One pass to compete it.\n"
    "//    Stage1:  caclaute each pixels\n"
    "//    Stage2:  Rediction whole workgroupsize into 1\n"
    "//    Local LDS = 64;\n"
    "\n"
    "#if 1\n"
    "void __kernel test_kernel(\n"
    "__global const float* filter,\n"
    "						__global const float* gradOutput,\n"
    "						__global float* images,\n"
    "						__global const float* dataBuf3,\n"
    "						__global const float* dataBuf4,\n"
    "        #ifdef BIASED\n"
    "             __global float *gradBiasWeights,\n"
    "				#else\n"
    "					 __global const float* dataBuf5,\n"
    "        #endif\n"
    "						__global float* gradWeights,\n"
    "						const int batchSize,\n"
    "						const int gOutputSize,\n"
    "						const int const3,\n"
    "						const int const4,\n"
    "						const int const5,\n"
    "						const int const6\n"
    " )\n"
    "\n"
    "#else\n"
    "void __kernel backprop_floats_fast(\n"
    "        __global const float *gradOutput,\n"
    "				__global const float *images,\n"
    "        __global float *gradWeights,\n"
    "        #ifdef BIASED\n"
    "             __global float *gradBiasWeights,\n"
    "        #endif\n"
    "				const float learningRateMultiplier,\n"
    "				const int batchSize,\n"
    "			  const int gOutputSize,\n"
    " )\n"
    " #endif\n"
    " {\n"
    "\n"
    "	 const float learningRateMultiplier = 0.0001f;\n"
    "	 int globalId = get_global_id(0);\n"
    "	 int localId  = get_local_id(0);\n"
    "	  float thiswchange = 0;\n"
    "	  __local float thiswchanges[FIXED_WORKGROUPSIZE];\n"
    "#ifdef BIASED\n"
    "	 thisbiaschange  = 0;\n"
    "#endif\n"
    "\n"
    "	 //It does not include any Invalid Threads since the FIXED_WORKGROUPSIZE is 64)\n"
    "   // if (globalId >= gNumFilters * gInputPlanes * gFilterSize * gFilterSize *  batchSize * FIXED_WORKGROUPSIZE) {\n"
    "   //     //Do nothing\n"
    "   // }\n"
    "	backprop_floats_fast_valid_thread(\n"
    "																					gradOutput,\n"
    "																					images,\n"
    "																					&thiswchange,\n"
    "																	#ifdef BIASED\n"
    "																					 &thisbiaschange,\n"
    "																	#endif\n"
    "																					batchSize,\n"
    "																					gOutputSize,\n"
    "																					globalId,\n"
    "																					localId\n"
    "\n"
    "		);\n"
    "\n"
    "\n"
    "		//aggregate Data to Thread0\n"
    "		Reduction_of_Weights(\n"
    "					&thiswchange,\n"
    "					thiswchanges,\n"
    "\n"
    "					#ifdef BIASED\n"
    "					&thisbiaschange,\n"
    "					thisbiaschanges,\n"
    "					#endif\n"
    "					localId,\n"
    "					globalId\n"
    "			);\n"
    "\n"
    "		//Thread0 Atomics into\n"
    "		if(localId == 0)\n"
    "		{\n"
    "					int globalIdOutput = (globalId >> FIXED_WORKGROUPSIZE_SHIFT) / batchSize ;\n"
    "\n"
    "						// gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]\n"
    "						//       aggregate over:  [outRow][outCol][n]\n"
    "						//gradWeights[ globalIdOutput ] = learningRateMultiplier * thiswchange;\n"
    "						volatile __global float *source = gradWeights + globalIdOutput;\n"
    "						AtomicAdd(source, learningRateMultiplier * thiswchange);\n"
    "#ifdef BIASED\n"
    "						int IntraFilterOffset =  globalIdOutput % gFilterSizeSquared;\n"
    "						int filterRow = IntraFilterOffset / gFilterSize;\n"
    "						int filterCol = IntraFilterOffset % gFilterSize;\n"
    "\n"
    "						int filter2Id = globalIdOutput / gFilterSizeSquared;\n"
    "						int outPlane = filter2Id / gInputPlanes;\n"
    "						int upstreamPlane = filter2Id % gInputPlanes;\n"
    "\n"
    "						bool writeBias = upstreamPlane == 0 && filterRow == gMargin && filterCol == gMargin;\n"
    "						if (writeBias) {\n"
    "								//gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;\n"
    "								*source = gradBiasWeights + outPlane;\n"
    "								AtomicAdd(source, learningRateMultiplier * thisbiaschange);\n"
    "						}\n"
    "#endif\n"
    "		}\n"
    "}\n"
    "\n"
    "//CNNBench command line\n"
    "//-dim 1 -lx 64 -ly 1 -gx 2359296 -gy 1 -f 3 -c1 128 -c2 128 -i 1 -x 4096 -y 4096\n"
    "	//    Input 	128x128, 4 Planes, BatchSize 128\n"
    "  //    output  128x128, 8 planes,  BatchSize 128\n"
    "  //    Filter Size = 3x3\n"
    "	//   numWorkgroups  = [outPlane][inputPlane][filterRow][filterCol] *  BatchSize\n"
    "  //   int globalSize = workgroupsize * numWorkgroups;\n"
    "	//    globalSize = 64 * [8][4][3][3] *  128   = 2359296\n"
    "  //\n"
    "	//    	    Input 	= 128x128x4x128 batch = 4096 * 2048\n"
    "  //			    output  = 128x128x8x128 batch = 4096 * 4096\n"
    "	//          force localthread_x =64,  globalthread_y =1\n"
    "	//          FilterSize =3,\n"
    "	//         -constant1 = batchSize 128\n"
    "	//         -constant2 = gOutputSize 128\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "test_kernel", options, "cl/fsword73_backpropweights_fast.cl");
    // [[[end]]]
}

