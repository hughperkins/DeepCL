#include "ConvolutionalLayer.h"
#include "NeuralNet.h"
#include "stringhelper.h"

ConvolutionalLayer::ConvolutionalLayer( Layer *previousLayer, ConvolutionalMaker const*maker ) :
        Layer( previousLayer, maker ),
        filterSize( maker->_filterSize ),
        filterSizeSquared( filterSize * filterSize ),
        padZeros( maker->_padZeros ),
        weightsWrapper( 0 ),
        resultsWrapper( 0 ),
        allocatedSpaceNumExamples( 0 ),
        resultsCopiedToHost( false ),
//        weightsCopiedToHost( false ),
        cl( maker->net->getCl() ) {
        if( padZeros && filterSize % 2 == 0 ) {
            throw std::runtime_error("filter size must be an odd number, if padZeros is true, so either turn off padZeros, or choose a different filtersize :-)");
        }
//        this->cl = new OpenCLHelper();
    if( filterSize > upstreamBoardSize ) {
            throw std::runtime_error("filter size cannot be larger than upstream board size: " + toString( filterSize) +
                " > " + toString(upstreamBoardSize) );
    }
    std::string options = "-D " + activationFunction->getDefineName();
    if( biased ) {
         options += " -D BIASED";
    }
//        const int batchSize, const int upstreamNumPlanes, const int numOutPlanes, 
//         const int upstreamBoardSize, const int filterSize, const int outBoardSize, const int padZeros, 
//    const int halfFilterSize = filterSize >> 1;
//    const int margin = gPadZeros ? halfFilterSize : 0;

    options += " -D gUpstreamBoardSize=" + toString(upstreamBoardSize);
    options += " -D gUpstreamBoardSizeSquared=" + toString(upstreamBoardSizeSquared);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gFilterSizeSquared=" + toString(filterSizeSquared);
    options += " -D gOutBoardSize=" + toString(boardSize);
    options += " -D gOutBoardSizeSquared=" + toString(boardSizeSquared);
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gNumOutPlanes=" + toString(numPlanes);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gHalfFilterSize=" + toString( filterSize >> 1 );
    options += " -D gUpstreamNumPlanes=" + toString(upstreamNumPlanes);
//    std::cout << "using kernel options: [" + options + "]" << std::endl;

//    options += " -D WORKGROUPSIZE 
    this->kernelConvolve = cl->buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2", options );
    this->kernelBackPropWeights = cl->buildKernel( "ClConvolve.cl", "backprop_floats", options );
//    this->kernelBackPropWeights2 = cl->buildKernel( "ClConvolve.cl", "backprop_floats_2", options );
//    this->kernelBackPropWeights3 = cl->buildKernel( "ClConvolve.cl", "backprop_floats_3", options );
//    this->kernelBackPropWeights4 = cl->buildKernel( "ClConvolve.cl", "backprop_floats_4", options );
    this->kernelBackPropWeightsWithScratch = cl->buildKernel( "ClConvolve.cl", "backprop_floats_withscratch", options );
    this->kernelBackpropErrors = cl->buildKernel( "ClConvolve.cl", "calcErrorsForUpstream", options );
    this->kernelBackpropBiasWeights = cl->buildKernel( "ClConvolve.cl", "doBiasBackprop", options );
    this->kernelAddInPlace = cl->buildKernel( "ClConvolve.cl", "add_in_place", options );
    this->kernelBackPropWeightsWithScratchAndBias = cl->buildKernel( "ClConvolve.cl", "backprop_floats_withscratch_dobias", options );
    biasWeights = new float[ getBiasWeightsSize() ];
    weights = new float[ getWeightsSize() ];
//    std::cout << " convolutional layer " << layerIndex << " allocating weights size " << getWeightsSize() << std::endl;
    randomizeWeights();
    weightsWrapper = cl->wrap( getWeightsSize(), weights );
    weightsWrapper->copyToDevice();
}


