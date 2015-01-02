#include "ConvolutionalLayer.h"
#include "NeuralNet.h"
#include "stringhelper.h"

ConvolutionalLayer::ConvolutionalLayer( Layer *previousLayer, ConvolutionalMaker const*maker ) :
        Layer( previousLayer, maker ),
        filterSize( maker->_filterSize ),
        padZeros( maker->_padZeros ),
        cl( maker->net->getCl() ) {
//        if( filterSize % 2 == 0 ) {
//            throw std::runtime_error("filter size must be an odd number");
//        }
//        this->cl = new OpenCLHelper();
    if( filterSize > upstreamBoardSize ) {
            throw std::runtime_error("filter size cannot be larger than upstream board size: " + toString( filterSize) +
                " > " + toString(upstreamBoardSize) );
    }
    std::string options = "-D " + activationFunction->getDefineName();
    if( biased ) {
         options += " -D BIASED";
    }
    this->kernelConvolve = cl->buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2", options );
    this->kernelBackPropWeights = cl->buildKernel( "ClConvolve.cl", "backprop_floats", options );
    this->kernelBackpropErrors = cl->buildKernel( "ClConvolve.cl", "calcErrorsForUpstream", options );
    biasWeights = new float[ getBiasWeightsSize() ];
    weights = new float[ getWeightsSize() ];
//    std::cout << " convolutional layer " << layerIndex << " allocating weights size " << getWeightsSize() << std::endl;
    randomizeWeights();
}


