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
using namespace easycl;

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
    // stringify.write_kernel2("kernel", "cl/BackpropWeightsScratchLarge.cl", "backprop_floats_withscratch_dobias_striped", 'options')
    // ]]]
    // generated using cog, from cl/BackpropWeightsScratchLarge.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2014,2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// expected defines:\n"
    "// BIASED (or not)\n"
    "\n"
    "// workgroupId: [outputPlane][inputPlane]\n"
    "// localId: [filterRow][filterCol]\n"
    "// per-thread iteration: [n][outputRow][outputCol]\n"
    "// local: errorimage: outputSize * outputSize\n"
    "//        imageimage: inputSize * inputSize\n"
    "// specific characteristic: load one stripe of each image at a time,\n"
    "// so we dont run out of memory\n"
    "// number of stripes set in: gNumStripes\n"
    "// note that whilst we can stripe the gradOutput simply,\n"
    "// we actually need to add a half-filter widthed additional few rows\n"
    "// onto the images stripe, otherwise we will be missing data\n"
    "//   we will call the size of the non-overlapping image stripes: gInputStripeInnerSize\n"
    "//      the outersize, including the two margins is: gInputStripeOuterSize\n"
    "//      of course, the first and last stripes will be missing a bit off the top/bottom, where the\n"
    "//      corresponding outer margin would be\n"
    "void kernel backprop_floats_withscratch_dobias_striped(\n"
    "        const float learningRateMultiplier, const int batchSize,\n"
    "         global const float *gradOutput, global const float *images,\n"
    "        global float *gradWeights,\n"
    "        #ifdef BIASED\n"
    "             global float *gradBiasWeights,\n"
    "        #endif\n"
    "        local float *_errorStripe, local float *_imageStripe\n"
    " ) {\n"
    "    // gHalfFilterSize\n"
    "    // gInputSize\n"
    "    //\n"
    "    // gInputStripeMarginRows => basically equal to gHalfFilterSize\n"
    "    // gInputStripeInnerNumRows = gInputSize / gNumStripes\n"
    "    // gInputStripeOuterNumRows = gInputStripeInnerNumRows + 2 * gHalfFilterSize  (note: one row less than\n"
    "    //                                                         if we just added gFilterSize)\n"
    "    // gInputStripeInnerSize = gInputStripeInnerNumRows * gInputSize\n"
    "    // gInputStripeOuterSize = gInputStripeOuterNumRows * gInputSize\n"
    "    // gInputStripeMarginSize = gInputStripeMarginRows * gInputSize\n"
    "    //\n"
    "    // gOutputStripeNumRows\n"
    "    // gOutputStripeSize\n"
    "\n"
    "    const int globalId = get_global_id(0);\n"
    "    const int localId = get_local_id(0);\n"
    "    const int workgroupId = get_group_id(0);\n"
    "    const int workgroupSize = get_local_size(0);\n"
    "\n"
    "    const int filterRow = localId / gFilterSize;\n"
    "    const int filterCol = localId % gFilterSize;\n"
    "\n"
    "    const int outPlane = workgroupId / gInputPlanes;\n"
    "    const int upstreamPlane = workgroupId % gInputPlanes;\n"
    "\n"
    "    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]\n"
    "    //       aggregate over:  [outRow][outCol][n]\n"
    "    float thiswchange = 0;\n"
    "#ifdef BIASED\n"
    "    float thisbiaschange = 0;\n"
    "#endif\n"
    "    const int numLoopsForImageStripe = (gInputStripeOuterSize + workgroupSize - 1) / workgroupSize;\n"
    "    const int numLoopsForErrorStripe = (gOutputSizeSquared + workgroupSize - 1) / workgroupSize;\n"
    "    for (int n = 0; n < batchSize; n++) {\n"
    "        const int imageImageGlobalOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;\n"
    "        const int imageImageGlobalOffsetAfter = imageImageGlobalOffset + gInputSizeSquared;\n"
    "        const int errorImageGlobalOffset = (n * gNumFilters + outPlane) * gOutputSizeSquared;\n"
    "        const int errorImageGlobalOffsetAfter = errorImageGlobalOffset + gOutputSizeSquared;\n"
    "        for (int stripe = 0; stripe < gNumStripes; stripe++) {\n"
    "            const int imageStripeInnerOffset = imageImageGlobalOffset + stripe * gInputStripeInnerSize;\n"
    "            const int imageStripeOuterOffset = imageStripeInnerOffset - gInputStripeMarginSize;\n"
    "            // need to fetch the image, but it's bigger than us, so will need to loop...\n"
    "            barrier(CLK_LOCAL_MEM_FENCE);\n"
    "            for (int i = 0; i < numLoopsForImageStripe; i++) {\n"
    "                int thisOffset = i * workgroupSize + localId;\n"
    "                int thisGlobalImagesOffset = imageStripeOuterOffset + thisOffset;\n"
    "                bool process = thisOffset < gInputStripeOuterSize\n"
    "                    && thisGlobalImagesOffset >= imageImageGlobalOffset\n"
    "                    && thisGlobalImagesOffset < imageImageGlobalOffsetAfter;\n"
    "                if (process) {\n"
    "                    _imageStripe[thisOffset] = images[ thisGlobalImagesOffset ];\n"
    "                }\n"
    "            }\n"
    "            int errorStripeOffset = errorImageGlobalOffset + stripe * gOutputStripeSize;\n"
    "            for (int i = 0; i < numLoopsForErrorStripe; i++) {\n"
    "                int thisOffset = i * workgroupSize + localId;\n"
    "                int globalErrorsOffset = errorStripeOffset + thisOffset;\n"
    "                bool process = thisOffset < gOutputStripeSize\n"
    "                    && globalErrorsOffset < errorImageGlobalOffsetAfter;\n"
    "                if (process) {\n"
    "                    _errorStripe[thisOffset ] = gradOutput[globalErrorsOffset];\n"
    "                }\n"
    "            }\n"
    "            const int stripeOutRowStart = stripe * gOutputStripeNumRows;\n"
    "            const int stripeOutRowEndExcl = stripeOutRowStart + gOutputStripeNumRows;\n"
    "            barrier(CLK_LOCAL_MEM_FENCE);\n"
    "//            if (localId == 13) {\n"
    "//                for (int i = 0; i < 12; i++) {\n"
    "//                    gradWeights[100 + stripe * 12 + i ] = _errorStripe[i * gOutputSize];\n"
    "//                }\n"
    "//                for (int i = 0; i < 20; i++) {\n"
    "//                    gradWeights[200 + stripe * 20 + i ] = _imageStripe[i * gInputSize];\n"
    "//                }\n"
    "//            }\n"
    "            if (localId < gFilterSizeSquared) {\n"
    "                for (int outRow = stripeOutRowStart; outRow < stripeOutRowEndExcl; outRow++) {\n"
    "                    int upstreamRow = outRow - gMargin + filterRow;\n"
    "                    for (int outCol = 0; outCol < gOutputSize; outCol++) {\n"
    "                        int upstreamCol = outCol - gMargin + filterCol;\n"
    "                        bool proceed =\n"
    "                            upstreamRow >= 0 && upstreamCol >= 0\n"
    "                            && upstreamRow < gInputSize && upstreamCol < gInputSize\n"
    "                            && outRow < gOutputSize;\n"
    "                        if (proceed) {\n"
    "                            int resultIndex = outRow * gOutputSize + outCol;\n"
    "                            float error = _errorStripe[resultIndex - stripe * gOutputStripeSize];\n"
    "                            int upstreamDataIndex = upstreamRow * gInputSize + upstreamCol;\n"
    "                            float upstreamResult = _imageStripe[upstreamDataIndex +  gInputStripeMarginSize\n"
    "                                        - stripe * gInputStripeInnerSize ];\n"
    "                            thiswchange += upstreamResult * error;\n"
    "        #ifdef BIASED\n"
    "                            thisbiaschange += error;\n"
    "        #endif\n"
    "                        }\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    if (localId < gFilterSizeSquared) {\n"
    "        gradWeights[ workgroupId * gFilterSizeSquared + localId ] = learningRateMultiplier * thiswchange;\n"
    "//        weightChanges[ workgroupId * gFilterSizeSquared + localId ] = workgroupId;\n"
    "    }\n"
    "#ifdef BIASED\n"
    "    bool writeBias = upstreamPlane == 0 && filterRow == gMargin && filterCol == gMargin;\n"
    "    if (writeBias) {\n"
    "        gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;\n"
    "    }\n"
    "#endif\n"
    "    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]\n"
    "    //       aggregate over:  [outRow][outCol][n]\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "backprop_floats_withscratch_dobias_striped", options, "cl/BackpropWeightsScratchLarge.cl");
    // [[[end]]]
}

