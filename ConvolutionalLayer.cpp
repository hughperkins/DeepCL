// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "ConvolutionalLayer.h"
#include "NeuralNet.h"
#include "stringhelper.h"

using namespace std;

ConvolutionalLayer::ConvolutionalLayer( Layer *previousLayer, ConvolutionalMaker const*maker ) :
        Layer( previousLayer, maker ),
        filterSize( maker->_filterSize ),
        filterSizeSquared( filterSize * filterSize ),
        padZeros( maker->_padZeros ),
        weightsWrapper( 0 ),
        resultsWrapper( 0 ),
//        errorsWrapper( 0 ),
        errorsForUpstreamWrapper( 0 ),
//        errors( 0 ),
        errorsForUpstream( 0 ),
        allocatedSpaceNumExamples( 0 ),
        resultsCopiedToHost( false ),
//        errorsCopiedToHost( false ),
        errorsForUpstreamCopiedToHost( false ),
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

// [virtual]
ConvolutionalLayer::~ConvolutionalLayer() {
    delete kernelBackPropWeights;
//        delete kernelBackPropWeights2;
//        delete kernelBackPropWeights3;
//        delete kernelBackPropWeights4;
    if( weightsWrapper != 0 ) {
        delete weightsWrapper;
    }
    if( resultsWrapper != 0 ) {
        delete resultsWrapper;
    }
//    if( errorsWrapper != 0 ) {
//        delete errorsWrapper;
//    }
//    if( errors != 0 ) {
//        delete[] errors;
//    }
    if( errorsForUpstreamWrapper != 0 ) {
        delete errorsForUpstreamWrapper;
    }
    if( errorsForUpstream != 0 ) {
        delete[] errorsForUpstream;
    }
    delete kernelBackPropWeightsWithScratch;
    delete kernelConvolve;
    delete kernelBackpropErrors;
    delete kernelBackpropBiasWeights;
    delete kernelAddInPlace;
    delete kernelBackPropWeightsWithScratchAndBias;
//        delete cl;
}
// [virtual]
float *ConvolutionalLayer::getErrorsForUpstream() {
    if( !errorsForUpstreamCopiedToHost ) {
        std::cout << "copying errorsForUpstream to host, from GPU" << std::endl;
        errorsForUpstreamWrapper->copyToHost();
        errorsForUpstreamCopiedToHost = true;
    }
    return errorsForUpstream;
}
// [virtual]
bool ConvolutionalLayer::providesErrorsWrapper() const {
    return true;
}
// [virtual]
CLWrapper *ConvolutionalLayer::getErrorsForUpstreamWrapper() {
    return errorsForUpstreamWrapper;
}
// [virtual]
void ConvolutionalLayer::initWeights( float*weights ) {
    Layer::initWeights( weights );
    weightsWrapper->copyToDevice();
}

// filters are organized like [filterid][plane][row][col]
void ConvolutionalLayer::randomizeWeights() {
//        std::cout << "convolutional layer randomzing weights" << std::endl;
    int fanin = upstreamNumPlanes * filterSize * filterSize;
    const int numThisLayerWeights = getWeightsSize();
    for( int i = 0; i < numThisLayerWeights; i++ ) {
        weights[i] = generateWeight( fanin );
    }
    for( int i = 0; i < numPlanes; i++ ) {
        biasWeights[i] = generateWeight( fanin );
    }
}
// [virtual]
bool ConvolutionalLayer::hasResultsWrapper() const {
    return true;
}
// [virtual]
CLWrapper *ConvolutionalLayer::getResultsWrapper() {
    return resultsWrapper;
}
// [virtual]
void ConvolutionalLayer::print() const {
    std::cout << "ConvolutionalLayer numFilters " << numPlanes << " filtersize " << filterSize << 
        " padZeros " << padZeros << " biased " << biased << " outputBoardSize " << boardSize << std::endl;
    printWeights();
    if( results != 0 ) {
        printOutput();
    }
}
// [virtual]
void ConvolutionalLayer::printWeights() const {
    std::cout << "  weights: " << std::endl;
// filters are organized like [filterid][plane][row][col]
    for( int filter = 0; filter < std::min( 5, numPlanes ); filter++ ) {
       std::cout << "    filter " << filter << std::endl;
       if( biased ) {
           std::cout << "       bias=" << biasWeights[filter] << std::endl;            
       }
       for( int plane = 0; plane < std::min(5,upstreamNumPlanes); plane++ ) {
           if( upstreamNumPlanes > 1 ) std::cout << "    inplane " << plane << std::endl;
            for( int i = 0; i < std::min(5,filterSize); i++ ) {
                std::cout << "      ";
                for( int j = 0; j < std::min(5,filterSize); j++ ) {
                   std::cout << getWeight( filter, plane, i, j ) << " ";
                }
                if( filterSize > 5 ) {
                   std::cout << " ...";
                }
                std::cout << std::endl;
            }
            if( filterSize > 5 ) {
               std::cout << " ..." << std::endl;
            }
        }
        if( upstreamNumPlanes > 5 ) std::cout << " ... other inplanes ... " << std::endl;
    }
    if( numPlanes > 5 ) std::cout << " ... other filters ... " << std::endl;
 }
 void ConvolutionalLayer::printOutput() const { // [virtual]
    if( results == 0 ) {
        return;
    }
    std::cout << "  outputs: " << std::endl;
// results are organized like [imageid][filterid][row][col]
    for( int n = 0; n < std::min( 5, batchSize ); n++ ) {
        std::cout << "    n: " << n << std::endl;
        for( int plane = 0; plane < std::min(5,numPlanes); plane++ ) {
            if( numPlanes > 1 ) std::cout << "      plane " << plane << std::endl;
            if( boardSize == 1 ) {
                 std::cout << "        " << getResult(n, plane, 0, 0 ) << std::endl;
            } else {
                for( int i = 0; i < std::min(5,boardSize); i++ ) {
                    std::cout << "      ";
                    for( int j = 0; j < std::min(5,boardSize); j++ ) {
                        std::cout << getResult( n, plane, i, j ) << " ";
                    }
                    if( boardSize > 5 ) std::cout << " ... ";
                    std::cout << std::endl;
                }
                if( boardSize > 5 ) std::cout << " ... " << std::endl;
            }
            if( numPlanes > 5 ) std::cout << " ... other planes ... " << std::endl;
        }
        if( batchSize > 5 ) std::cout << " ... other n ... " << std::endl;
    }
}
// [virtual]
void ConvolutionalLayer::setBatchSize( int batchSize ) {
    if( batchSize <= allocatedSpaceNumExamples ) {
        this->batchSize = batchSize;
        return;
    }
    if( results != 0 ) {
        delete[] results;
    }
    if( resultsWrapper != 0 ) {
        delete resultsWrapper;
    }
    if( errorsForUpstream != 0 ) {
        delete[] errorsForUpstream;
    }
    if( errorsForUpstreamWrapper != 0 ) {
        delete errorsForUpstreamWrapper;
    }
    this->batchSize = batchSize;
    results = new float[getResultsSize()];
    resultsWrapper = cl->wrap( getResultsSize(), results );
//        std::cout << " layer " << layerIndex << " allocating results size " << getResultsSize() << std::endl;
    weOwnResults = true;
    if( layerIndex > 1 ) {
        errorsForUpstream = new float[ previousLayer->getResultsSize() ];
        errorsForUpstreamWrapper = cl->wrap( previousLayer->getResultsSize(), errorsForUpstream );
    }
    this->allocatedSpaceNumExamples = batchSize;
}
// [virtual]
void ConvolutionalLayer::propagate() {
    StatefulTimer::instance()->timeCheck("    propagate layer " + toString( layerIndex ) + ", START");

    CLWrapper *upstreamWrapper = 0;
    if( previousLayer->hasResultsWrapper() ) {
//            std::cout << "layer " << previousLayer->layerIndex << " has resultsWrapper" << std::endl;
        upstreamWrapper = previousLayer->getResultsWrapper();
    } else {
//            std::cout << "layer " << previousLayer->layerIndex << " has no resultsWrapper" << std::endl;
        upstreamWrapper = cl->wrap( previousLayer->getResultsSize(), (float *)previousLayer->getResults() );
        upstreamWrapper->copyToDevice();
    }

//        timer.timeCheck("    propagate, copied to device");

    CLFloatWrapper *biasWeightsWrapper = 0;
    if( biased ) {
        biasWeightsWrapper = cl->wrap( getBiasWeightsSize(), biasWeights );
        biasWeightsWrapper->copyToDevice();
    }
    StatefulTimer::instance()->timeCheck("    propagate layer " + toString( layerIndex ) + ", copied to device");
    kernelConvolve->in(batchSize)->in( upstreamNumPlanes )->in( numPlanes )->in( upstreamBoardSize )->in( filterSize )
      ->in( padZeros ? 1 : 0 );
    kernelConvolve->input( upstreamWrapper );
    kernelConvolve->input( weightsWrapper);
    if( biased ) {
        kernelConvolve->input( biasWeightsWrapper);
    }
    kernelConvolve->output( resultsWrapper );
    int globalSize = getResultsSize();
//        std::cout << "requested globalsize: " << globalSize << std::endl;
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//        timer.timeCheck("    propagate, passed in inputs");
//        std::cout << "globalsize " << globalSize << " workgroupsize " << workgroupsize <<
//           " upsteramwrappersize " << upstreamWrapper->size() << std::endl;
    kernelConvolve->run_1d( globalSize, workgroupsize );
    cl->finish();
//        resultsWrapper->copyToHost();
    StatefulTimer::instance()->timeCheck("    propagate layer " + toString( layerIndex ) + ",  after clFinish");
    // if we are the last layer, then copy results to host

    if( !previousLayer->hasResultsWrapper() ) {
        delete upstreamWrapper;
    }
    if( biased ) {
        delete biasWeightsWrapper;
    }
    resultsCopiedToHost = false;
}
// [virtual]
float * ConvolutionalLayer::getResults() {
    if( !resultsCopiedToHost ) {
//            std::cout << "layer " << layerIndex << " copying results to host " << std::endl;
        resultsWrapper->copyToHost();
        resultsCopiedToHost = true;
    }
    return results;
};
// [virtual]
int ConvolutionalLayer::getWeightsSize() const {
    return numPlanes * upstreamNumPlanes * filterSize * filterSize;
}
// [virtual]
int ConvolutionalLayer::getBiasWeightsSize() const {
    return numPlanes;
}
//// [virtual]
//float *getErrors() {
//    if( !errorsOnHost ) {
//        errorsWrapper.copyToHost();
//        errorsOnHost = true;
//    }
//    return errors;
//}

// weights:     [outPlane][upstreamPlane][filterRow][filterCol]
//       aggregate over:  [outRow][outCol][n]
// biasweights: [outPlane]
//       aggregate over:  [upstreamPlane][filterRow][filterCol][outRow][outCol][n]

// [virtual]
void ConvolutionalLayer::backPropErrors( float learningRate ) {
//        Timer timer;
    float *weightChanges = new float[ getWeightsSize() ];
    float *biasWeightChanges = new float[getBiasWeightsSize()];

    CLWrapper *weightChangesWrapper = cl->wrap( getWeightsSize(), weightChanges );

    StatefulTimer::instance()->timeCheck("backproperrors(): start backprop, layer " + toString( layerIndex ) );

    CLWrapper *imagesWrapper = 0;
    if( previousLayer->hasResultsWrapper() ) {
        imagesWrapper = previousLayer->getResultsWrapper();
    } else {
        imagesWrapper = cl->wrap( previousLayer->getResultsSize(), previousLayer->getResults() );
        imagesWrapper->copyToDevice();
    }

    CLWrapper *errorsWrapper = 0;
    bool weOwnErrorsWrapper = false;
    if( nextLayer->providesErrorsWrapper() ) {
        errorsWrapper = nextLayer->getErrorsForUpstreamWrapper();
    } else {
        errorsWrapper = cl->wrap( getResultsSize(), nextLayer->getErrorsForUpstream() );
        errorsWrapper->copyToDevice();
        weOwnErrorsWrapper = true;
    }

    bool implicitlyCalcedBiasWeight = false;
    if( filterSize <= 19 ) {
        backPropWeightsGpuWithScratchAndBias( learningRate, imagesWrapper, resultsWrapper, errorsWrapper, weightChangesWrapper, biasWeightChanges );
        implicitlyCalcedBiasWeight = true;
    } else {
        backPropWeightsGpu( learningRate, imagesWrapper, resultsWrapper, errorsWrapper, weightChangesWrapper );
    }
    StatefulTimer::instance()->timeCheck("backproperrors(): done weight backprop, layer " + toString( layerIndex ) );
    if( !implicitlyCalcedBiasWeight ) {
        doBiasBackpropGpu( learningRate, resultsWrapper, errorsWrapper, biasWeightChanges );
    StatefulTimer::instance()->timeCheck("backproperrors(): done biasweight backprop, layer " + toString( layerIndex ) );
    }

    if( layerIndex > 1 ) {
        calcErrorsForUpstreamGpu( weightsWrapper, errorsWrapper, errorsForUpstreamWrapper );
        StatefulTimer::instance()->timeCheck("backproperrors(): calced errors for upstream, layer " + toString( layerIndex ) );
    }

    updateWeightsGpu( weightChangesWrapper, weightsWrapper );
        StatefulTimer::instance()->timeCheck("backproperrors(): updated weights, layer " + toString( layerIndex ) );

    const int numWeights = getWeightsSize();
    for( int plane = 0; plane < numPlanes; plane++ ) {
        biasWeights[plane] += biasWeightChanges[plane];
    }

    delete[] biasWeightChanges;
    if( !previousLayer->hasResultsWrapper() ) {
        delete imagesWrapper;
    }
    delete weightChangesWrapper;
    delete[] weightChanges;
    if( weOwnErrorsWrapper ) {
        delete errorsWrapper;
    }
}

void ConvolutionalLayer::updateWeightsGpu( CLWrapper* weightChangesWrapper, CLWrapper*weightsWrapper ) {
    int globalSize = getWeightsSize();
    int workgroupsize = std::min( getWeightsSize(), cl->getMaxWorkgroupSize() );
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    kernelAddInPlace->in( getWeightsSize() )
        ->in( weightChangesWrapper )
        ->in (weightsWrapper)
        ->run_1d( globalSize, workgroupsize );
    cl->finish();
}

void ConvolutionalLayer::backPropWeightsCpu( float learningRate, float const *errors, float *weights ) {
//        Timer timer;
    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
//        const bool debug = false;
    const int halfFilterSize = filterSize >> 1;
    const int margin = padZeros ? halfFilterSize : 0;
    StatefulTimer::instance()->timeCheck(" backpropweightscpu start, layer " + toString( layerIndex ) );
    for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
        for( int upstreamPlane = 0; upstreamPlane < upstreamNumPlanes; upstreamPlane++ ) {
            for( int filterRow = 0; filterRow < filterSize; filterRow++ ) {
                for( int filterCol = 0; filterCol < filterSize; filterCol++ ) {
                    int weightIndex = getWeightIndex( outPlane, upstreamPlane, filterRow, filterCol );
//                        if( filterRow != 1 || filterCol > 1 ) {
//                            weights[weightIndex] = 0;
//                            continue;
//                        }
                    float thiswchange = 0;
                    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
                    //       aggregate over:  [outRow][outCol][n]
                    for( int outRow = 0; outRow < boardSize; outRow++ ) {
                        int upstreamRow = outRow - margin + filterRow;
                        for( int outCol = 0; outCol < boardSize; outCol++ ) {
                            int upstreamCol = outCol - margin + filterCol;
                            for( int n = 0; n < batchSize; n++ ) {
                                int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                                float error = errors[resultIndex];
                                float actualOutput = results[resultIndex];
                                float activationDerivative = activationFunction->calcDerivative( actualOutput );
//                                    float activationDerivative = 1 - actualOutput * actualOutput;
                                float upstreamResult = previousLayer->getResult( n, upstreamPlane, upstreamRow, upstreamCol );
                                float thisimagethiswchange = upstreamResult * activationDerivative *
                                error;
                                thiswchange += thisimagethiswchange;
//    if(debug)std::cout << "outPlane=" << outPlane << " inPlane=" << upstreamPlane << " filterpos=" << filterRow << "," << filterCol
//       << " outpos=" << outRow << "," << outCol << " n=" << n << " resindex " << resultIndex << " error=" << error
//       << " actualoutput=" << actualOutput << " upstreampos=" << upstreamRow <<"," << upstreamCol << " upstreamResult=" << upstreamResult << " thisimagethiswchange="
//       << thisimagethiswchange << std::endl;
                            }
                        }
                    }
//                        weights[ weightIndex ] -= learningRate * thiswchange / batchSize / sqrt( boardSize * boardSize );
                    weights[ weightIndex ] += - thiswchange * learningMultiplier;
                }
            }
        }
    }
//        timer.timeCheck("did backprop to ourselves v2");
    StatefulTimer::instance()->timeCheck(" backpropweightscpu end, layer " + toString( layerIndex ) );
}

void ConvolutionalLayer::backPropWeightsGpu( float learningRate, CLWrapper *imagesWrapper, CLWrapper *resultsWrapper, CLWrapper*errorsWrapper, CLWrapper *weightChangesWrapper ) {
    StatefulTimer::instance()->timeCheck(" backpropweightsGpu start, layer " + toString( layerIndex ) );
    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
//    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), (float *)errors );
//    errorsWrapper->copyToDevice();
    kernelBackPropWeights
       ->in(learningMultiplier)
       ->in( batchSize )->in( upstreamNumPlanes )->in(numPlanes)
       ->in( upstreamBoardSize )->in( filterSize )->in( boardSize )->in( padZeros ? 1 : 0 )
       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper );
    int globalSize = getWeightsSize();
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//        std::cout << " backpropgpu, globalsize " << globalSize << " workgroupsize " << workgroupsize << std::endl;
    kernelBackPropWeights->run_1d(globalSize, workgroupsize);

    cl->finish();

//        timer.timeCheck("backPropGpu");
//    delete errorsWrapper;
    StatefulTimer::instance()->timeCheck(" backpropweightsGpu end, layer " + toString( layerIndex ) );
}

void ConvolutionalLayer::backPropWeightsGpuWithScratch( float learningRate, CLWrapper *imagesWrapper, CLWrapper *resultsWrapper, CLWrapper*errorsWrapper, CLWrapper *weightChangesWrapper ) {
//        Timer timer;
    StatefulTimer::instance()->timeCheck(" backpropweightsGpuWithScratch start, layer " + toString( layerIndex ) );
//        int globalSize = getWeightsSize();
    int workgroupsize = filterSizeSquared;
    int numWorkgroups = upstreamNumPlanes * numPlanes;
    int globalSize = workgroupsize * numWorkgroups;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//        std::cout << " backpropgpuwithscratch, globalsize " << globalSize << " workgroupsize " << workgroupsize << std::endl;

    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
//    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), (float *)errors );
//        imagesWrapper->copyToDevice();
//    errorsWrapper->copyToDevice();
    kernelBackPropWeightsWithScratch
       ->in(learningMultiplier)
       ->in( batchSize )
       
        ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )

        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared );
    kernelBackPropWeightsWithScratch->run_1d(globalSize, workgroupsize);

    cl->finish();

//        timer.timeCheck("backPropGpu");
//    delete errorsWrapper;
//        delete weightChangesWrapper;
    StatefulTimer::instance()->timeCheck(" backpropweightsGpuWithScratch end, layer " + toString( layerIndex ) );
}

void ConvolutionalLayer::backPropWeightsGpuWithScratchAndBias( float learningRate, CLWrapper *imagesWrapper, CLWrapper *resultsWrapper, CLWrapper *errorsWrapper, CLWrapper *weightChangesWrapper, float *biasWeightChanges ) {
    StatefulTimer::instance()->timeCheck(" backpropweightsGpuWithScratchAndBias start, layer " + toString( layerIndex ) );
    int workgroupsize = filterSizeSquared;
    int numWorkgroups = upstreamNumPlanes * numPlanes;
    int globalSize = workgroupsize * numWorkgroups;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;

    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
//    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), (float *)errors );
    CLWrapper *biasWeightChangesWrapper = cl->wrap( numPlanes, biasWeightChanges );
//    errorsWrapper->copyToDevice();
    kernelBackPropWeightsWithScratchAndBias
       ->in(learningMultiplier)
       ->in( batchSize )
       
        ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        ->out( biasWeightChangesWrapper )

        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared );
    kernelBackPropWeightsWithScratchAndBias->run_1d(globalSize, workgroupsize);

    cl->finish();
    biasWeightChangesWrapper->copyToHost();

//    delete errorsWrapper;
    delete biasWeightChangesWrapper;
    StatefulTimer::instance()->timeCheck(" backpropweightsGpuWithScratchAndBias end, layer " + toString( layerIndex ) );
}
/*
void backPropWeightsGpu2( float learningRate, float const*errors, float *weightChanges ) {
    // soooo.... going to feed in same data as before, but structure workgroups differently...

//        void kernel backprop_floats_2( const float learningRateMultiplier,
//        const int batchSize, const int upstreamNumPlanes, const int numOutPlanes, 
//         const int upstreamBoardSize, const int filterSize, const int outBoardSize, const int padZeros, 
//         global const float *upstreamBoardsGlobal, 
//         global const float *resultsGlobal, global const float *errorsGlobal,
//         global float *weightChangesGlobal ) {

    int globalSize = getWeightsSize();
//        int workgroupsize = cl->getMaxWorkgroupSize();
    int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
    std::cout << " workgroupsize " << workgroupsize << " globalsize " << globalSize << std::endl;

    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
    CLWrapper *imagesWrapper = cl->wrap( previousLayer->getResultsSize(), previousLayer->getResults() );
    CLWrapper *resultsWrapper = cl->wrap( getResultsSize(), results );
    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), errors );
    CLWrapper *weightChangesWrapper = cl->wrap( getWeightsSize(), weightChanges );
    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();
    kernelBackPropWeights2
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl->getNextPower2( workgroupsize ) )
//->in( upstreamNumPlanes )->in(numPlanes)
//           ->in( upstreamBoardSize )->in( filterSize )->in( boardSize )->in( padZeros ? 1 : 0 )

       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( workgroupsize );

    kernelBackPropWeights2->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();
//        cl->finish();

//        timer.timeCheck("backPropGpu");
    delete imagesWrapper;
    delete resultsWrapper;
    delete errorsWrapper;
    delete weightChangesWrapper;
}
*/
/*
void backPropWeightsGpu3( const float learningRate, float const*const errors, float *const weightChanges ) {
    // each workgroup is dimensioned to be big enough to loop over the usptream Board
    // round to nearest 32, which about fills an average compute unit (16 or 32)
    const int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    // then, once we have the workgroup size, well, first how many workgroups?
    // it is: number outplanes * number inplanes:
    const int numWorkgroups = upstreamNumPlanes * numPlanes;
    //multiply, to get globalsize:
    const int globalSize = numWorkgroups * workgroupsize;
    std::cout << " workgroupsize " << workgroupsize << " globalsize " << globalSize << std::endl;
    // yay :-)

    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
    CLWrapper *imagesWrapper = cl->wrap( previousLayer->getResultsSize(), previousLayer->getResults() );
    CLWrapper *resultsWrapper = cl->wrap( getResultsSize(), results );
    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), errors );
    CLWrapper *weightChangesWrapper = cl->wrap( getWeightsSize(), weightChanges );

    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();
    kernelBackPropWeights3
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl->getNextPower2( workgroupsize ) )

       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared )
        ->localFloats( workgroupsize );

    kernelBackPropWeights3->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();

//        // reduce on cpu for now :-)
//        // need to reduce over ...
//        for( int filterId = 0; filterId < numPlanes; filterId++ ) {
//            for( int filterPos = 0; filterPos < filterBoardSizeSquared; filterPos++ ) {
//                float sum = 0;
//                for( int 
////                for( int workgroupId = 0; workgroupId < numWorkgroups; workgroupId++ ) {
//                    sum += weightChangesReduceArea[  ];
//                }
//                weightChanges[ filterId * filterBoardSizeSquared + filterPos ] = sum;
//            }
//        }

//        cl->finish();

//        timer.timeCheck("backPropGpu");
    delete imagesWrapper;
    delete resultsWrapper;
    delete errorsWrapper;
    delete weightChangesWrapper;
}
*/
/*
void backPropWeightsGpu4( const float learningRate, float const*const errors, float *const weightChanges ) {
    // each workgroup is dimensioned to be big enough to loop over the usptream Board
    // round to nearest 32, which about fills an average compute unit (16 or 32)
    const int workgroupsize = ( ( upstreamBoardSizeSquared + 31 ) / 32 ) * 32;
    // then, once we have the workgroup size, well, first how many workgroups?
    // it is: number outplanes * number inplanes:
    const int numWorkgroups = numPlanes;
    //multiply, to get globalsize:
    const int globalSize = numWorkgroups * workgroupsize;
    std::cout << " workgroupsize " << workgroupsize << " globalsize " << globalSize << std::endl;
    // yay :-)

    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
    CLWrapper *imagesWrapper = cl->wrap( previousLayer->getResultsSize(), previousLayer->getResults() );
    CLWrapper *resultsWrapper = cl->wrap( getResultsSize(), results );
    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), errors );
    CLWrapper *weightChangesWrapper = cl->wrap( getWeightsSize(), weightChanges );

    imagesWrapper->copyToDevice();
    resultsWrapper->copyToDevice();
    errorsWrapper->copyToDevice();
    kernelBackPropWeights4
       ->in(learningMultiplier)
       ->in( batchSize )
        ->in( cl->getNextPower2( workgroupsize ) )

       ->in( imagesWrapper )
       ->in(resultsWrapper)
       ->in( errorsWrapper )
       ->out( weightChangesWrapper )
        
        ->localFloats( upstreamBoardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( boardSizeSquared )
        ->localFloats( filterSizeSquared * upstreamNumPlanes )
        ->localFloats( workgroupsize );

    kernelBackPropWeights4->run_1d(globalSize, workgroupsize);

    weightChangesWrapper->copyToHost();

//        // reduce on cpu for now :-)
//        // need to reduce over ...
//        for( int filterId = 0; filterId < numPlanes; filterId++ ) {
//            for( int filterPos = 0; filterPos < filterBoardSizeSquared; filterPos++ ) {
//                float sum = 0;
//                for( int 
////                for( int workgroupId = 0; workgroupId < numWorkgroups; workgroupId++ ) {
//                    sum += weightChangesReduceArea[  ];
//                }
//                weightChanges[ filterId * filterBoardSizeSquared + filterPos ] = sum;
//            }
//        }

//        cl->finish();

//        timer.timeCheck("backPropGpu");
    delete imagesWrapper;
    delete resultsWrapper;
    delete errorsWrapper;
    delete weightChangesWrapper;
}
*/

//// [virtual]
//bool ConvolutionalLayer::needErrorsBackprop() {
//    return true;
//}

void ConvolutionalLayer::calcErrorsForUpstreamGpu( CLWrapper *weightsWrapper, CLWrapper *errorsWrapper, CLWrapper *errorsForUpstreamWrapper ) {
//        CLWrapper *weightsWrapper = cl->wrap( getWeightsSize(), weights );
    StatefulTimer::instance()->timeCheck("calcErrorsForUpstreamGpu start, layer " + toString( layerIndex ) );
//    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), (float *)errors );
//    CLWrapper *errorsForUpstreamWrapper = cl->wrap( previousLayer->getResultsSize(), errorsForUpstream );
//        weightsWrapper->copyToDevice();
//    errorsWrapper->copyToDevice();
    kernelBackpropErrors
        ->in( upstreamNumPlanes )->in( upstreamBoardSize )->in( filterSize )
        ->in( numPlanes )->in( boardSize )
        ->in( padZeros ? 1 : 0 )
        ->in( weightsWrapper )
        ->in( errorsWrapper )
        ->out( errorsForUpstreamWrapper );
    int globalSize = previousLayer->getResultsSize();
    int workgroupsize = cl->getMaxWorkgroupSize();
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//        std::cout << "calcerrorsforupstreamgpu workgroupsize " << workgroupsize << " globalsize " << globalSize << std::endl;
    kernelBackpropErrors->run_1d(globalSize, workgroupsize);

    cl->finish();
//    errorsForUpstreamWrapper->copyToHost();

//        StatefulTimer::instance()->timeCheck("    calcErrorsForUpstreamGpu, finished kernel, layer " + toString( layerIndex ) );
//        StatefulTimer::instance()->timeCheck("    calcErrorsForUpstreamGpu, copied results to host, layer " + toString( layerIndex ) );
//    delete errorsForUpstreamWrapper;
//    delete errorsWrapper;
//        delete weightsWrapper;
    StatefulTimer::instance()->timeCheck("calcErrorsForUpstreamGpu end, layer " + toString( layerIndex ) );
}

void ConvolutionalLayer::calcErrorsForUpstreamCpu( float const *const weights, float const *const errors, float *errorsForUpstream ) {
//        Timer timer;
    StatefulTimer::instance()->timeCheck("calcErrorsForUpstreamCpu start, layer " + toString( layerIndex ) );
    const int halfFilterSize = filterSize >> 1;
    const int margin = padZeros ? halfFilterSize : 0;
    // handle lower layer...
    // errors for upstream look like [n][inPlane][inRow][inCol]
    // need to aggregate over: [outPlane][outRow][outCol] (?)
    // need to backprop errors along each possible weight
    // each upstream feeds to:
    //    - each of our filters (so numPlanes filters)
    //    - each of our outpoint points (so boardSize * boardSize)
    // for our own backprop, we updated weights for:
    //      [outPlane][inPlane][filterRow][filtercol]
    //    aggregating over: [n][outRow][outCol]
    // errors are provider per [n][inPlane][inRow][inCol]
    for( int n = 0; n < batchSize; n++ ) {
        for( int upstreamPlane = 0; upstreamPlane < upstreamNumPlanes; upstreamPlane++ ) {
            for( int upstreamRow = 0; upstreamRow < upstreamBoardSize; upstreamRow++ ) {
                int minFilterRow = std::max( 0, upstreamRow + margin - (boardSize - 1) );
                int maxFilterRow = std::min( filterSize - 1, upstreamRow + margin );
                for( int upstreamCol = 0; upstreamCol < upstreamBoardSize; upstreamCol++ ) {
                    float sumWeightTimesOutError = 0;
                    // aggregate over [outPlane][outRow][outCol]
                    int minFilterCol = std::max( 0, upstreamCol + margin - (boardSize -1) );
                    int maxFilterCol = std::min( filterSize - 1, upstreamCol + margin );
                    for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
                        for( int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++ ) {
                            int outRow = upstreamRow + margin - filterRow;
                            for( int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++ ) {
                                int outCol = upstreamCol + margin - filterCol;
                                int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                                float thisError = errors[resultIndex];
                                int thisWeightIndex = getWeightIndex( outPlane, upstreamPlane, filterRow, filterCol );
                                float thisWeight = weights[thisWeightIndex];
                                float thisWeightTimesError = thisWeight * thisError;
                                sumWeightTimesOutError += thisWeightTimesError;
                            }
                        }
                    }
                    int upstreamResultIndex = previousLayer->getResultIndex( n, upstreamPlane, upstreamRow, upstreamCol );
                    errorsForUpstream[upstreamResultIndex] = sumWeightTimesOutError;
                }
            }
        }
    }
//        timer.timeCheck("calced errors for upstream");   
    StatefulTimer::instance()->timeCheck("calcErrorsForUpstreamCpu end, layer " + toString( layerIndex ) );
}

void ConvolutionalLayer::doBiasBackpropCpu(float learningRate, float const *results, float const *errors, float *biasWeightChanges ) {
//        Timer timer;
    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
    const bool debug = false;
    if( !biased ) {
         return;
     }
    StatefulTimer::instance()->timeCheck("doBiasBackpropCpu start, layer " + toString( layerIndex ) );
     for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
        // bias...
        // biasweights: [outPlane]
        //       aggregate over:  [upstreamPlane][filterRow][filterCol][outRow][outCol][n]
        float thiswchange = 0;
        for( int n = 0; n < batchSize; n++ ) {
            for( int outRow = 0; outRow < boardSize; outRow++ ) {
                for( int outCol = 0; outCol < boardSize; outCol++ ) {
                    float upstreamResult = 1;
                    int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                    float actualOutput = results[resultIndex];
                    float activationDerivative = activationFunction->calcDerivative( actualOutput );
                    float thisimagethiswchange = upstreamResult * errors[resultIndex] * activationDerivative;
                    thiswchange += thisimagethiswchange;
if(debug)std::cout << "bias outPlane=" << outPlane << " outpos=" << outRow << "," << outCol << " n=" << n << " resindex " << resultIndex << " error=" << errors[resultIndex]
   << " actualoutput=" << actualOutput << " upstreamResult=" << upstreamResult << " thisimagethiswchange="
   << thisimagethiswchange << std::endl;
                }
            }
        }
        biasWeightChanges[ outPlane ] = - learningMultiplier * thiswchange;
     }
//        timer.timeCheck("did bias backprop");   
    StatefulTimer::instance()->timeCheck("doBiasBackpropCpu end, layer " + toString( layerIndex ) );
}

void ConvolutionalLayer::doBiasBackpropGpu(float learningRate, CLWrapper *resultsWrapper, CLWrapper *errorsWrapper, float *biasWeightChanges ) {
//        Timer timer;
    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
    if( !biased ) {
         return;
    }
    StatefulTimer::instance()->timeCheck("doBiasBackpropGpu start, layer " + toString( layerIndex ) );
    int globalSize = numPlanes;
    int workgroupsize = std::min( numPlanes, cl->getMaxWorkgroupSize() );
    globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;

//        CLWrapper *resultsWrapper = cl->wrap( getResultsSize(), results );
//    CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), (float *)errors );
    CLWrapper *biasWeightChangesWrapper = cl->wrap( numPlanes, biasWeightChanges );
//        resultsWrapper->copyToDevice();
//    errorsWrapper->copyToDevice();

    kernelBackpropBiasWeights->in( learningMultiplier )->in( batchSize )
        ->in( resultsWrapper )->in( errorsWrapper )->out( biasWeightChangesWrapper );
    kernelBackpropBiasWeights->run_1d(globalSize, workgroupsize);
    cl->finish();
    biasWeightChangesWrapper->copyToHost();

    delete biasWeightChangesWrapper;
//        delete resultsWrapper;
//    delete errorsWrapper;
    StatefulTimer::instance()->timeCheck("doBiasBackpropGpu end, layer " + toString( layerIndex ) );
}

