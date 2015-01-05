// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "OpenCLHelper.h"
//#include "ClConvolve2.h"
#include "ActivationFunction.h"
#include "LayerMaker.h"
#include "Timer.h"
#include "StatefulTimer.h"
#include "stringhelper.h"

class ConvolutionalLayer : public Layer {
public:
    OpenCLHelper *const cl; // NOT owned by us
    CLKernel *kernelConvolve;
    CLKernel *kernelBackPropWeights;
//    CLKernel *kernelBackPropWeights2;
//    CLKernel *kernelBackPropWeights3;
//    CLKernel *kernelBackPropWeights4;
    CLKernel *kernelBackPropWeightsWithScratch;
    CLKernel *kernelBackPropWeightsWithScratchAndBias;
    CLKernel *kernelBackpropErrors;
    CLKernel *kernelBackpropBiasWeights;
    CLKernel *kernelAddInPlace;

    const int filterSize;
    const int filterSizeSquared;
    const bool padZeros;

    CLWrapper *weightsWrapper;
    CLWrapper *resultsWrapper;

    int allocatedSpaceNumExamples;

    bool resultsCopiedToHost;
//    bool weightsCopiedToHost;

    ConvolutionalLayer( Layer *previousLayer, ConvolutionalMaker const*maker );

    virtual ~ConvolutionalLayer() {
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
        delete kernelBackPropWeightsWithScratch;
        delete kernelConvolve;
        delete kernelBackpropErrors;
        delete kernelBackpropBiasWeights;
        delete kernelAddInPlace;
        delete kernelBackPropWeightsWithScratchAndBias;
//        delete cl;
    }
    virtual void initWeights( float*weights ) {
        Layer::initWeights( weights );
        weightsWrapper->copyToDevice();
    }

// filters are organized like [filterid][plane][row][col]
    void randomizeWeights() {
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
    virtual bool hasResultsWrapper() const {
        return true;
    }
    virtual CLWrapper *getResultsWrapper() {
        return resultsWrapper;
    }
    virtual void print() const {
        std::cout << "ConvolutionalLayer numFilters " << numPlanes << " filtersize " << filterSize << 
            " padZeros " << padZeros << " biased " << biased << " outputBoardSize " << boardSize << std::endl;
        printWeights();
        if( results != 0 ) {
            printOutput();
        }
    }
    virtual void printWeights() const {
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
     virtual void printOutput() const {
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
    virtual void setBatchSize( int batchSize ) {
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
        this->batchSize = batchSize;
        results = new float[getResultsSize()];
        resultsWrapper = cl->wrap( getResultsSize(), results );
//        std::cout << " layer " << layerIndex << " allocating results size " << getResultsSize() << std::endl;
        weOwnResults = true;
        this->allocatedSpaceNumExamples = batchSize;
    }
    virtual void propagate() {
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
    virtual float * getResults() {
        if( !resultsCopiedToHost ) {
//            std::cout << "layer " << layerIndex << " copying results to host " << std::endl;
            resultsWrapper->copyToHost();
            resultsCopiedToHost = true;
        }
        return results;
    };
    virtual int getWeightsSize() const {
        return numPlanes * upstreamNumPlanes * filterSize * filterSize;
    }
    virtual int getBiasWeightsSize() const {
        return numPlanes;
    }
    // images are organized like [imageId][plane][boardrow][boardcol]
    // filters are organized like [filterid][plane][filterrow][filtercol]
    // results are organized like [imageid][filterid][boardrow][boardcol]
    inline int getWeightIndex( int outPlane, int inPlane, int filterrow, int filtercol ) const {
        return ( ( outPlane * upstreamNumPlanes 
             + inPlane ) * filterSize 
             + filterrow ) * filterSize
             + filtercol;
    }
    inline float getWeight( int outPlane, int inPlane, int filterrow, int filtercol ) const {
        return weights[getWeightIndex( outPlane, inPlane, filterrow, filtercol ) ];
    }
    virtual void calcErrors( float const *expected, float *errors ) {
//        Timer timer;
        getResults();
        // matrix per-element subtraction...
        for( int n = 0; n < batchSize; n++ ) {
            for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
                for( int outRow = 0; outRow < boardSize; outRow++ ) {
                    for( int outCol = 0; outCol < boardSize; outCol++ ) {
                        int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                        errors[ resultIndex ] = results[resultIndex] - expected[resultIndex];
                    }
                } 
            }
        }
//        timer.timeCheck("expected->errors done");
//        backPropErrors( learningRate, errors );
//        delete[] errors;
    }

    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    // biasweights: [outPlane]
    //       aggregate over:  [upstreamPlane][filterRow][filterCol][outRow][outCol][n]

    virtual void backPropErrors( float learningRate, float const *errors, float *errorsForUpstream ) {
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

        bool implicitlyCalcedBiasWeight = false;
        if( filterSize <= 19 ) {
            backPropWeightsGpuWithScratchAndBias( learningRate, imagesWrapper, resultsWrapper, errors, weightChangesWrapper, biasWeightChanges );
            implicitlyCalcedBiasWeight = true;
        } else {
            backPropWeightsGpu( learningRate, imagesWrapper, resultsWrapper, errors, weightChangesWrapper );
        }
        StatefulTimer::instance()->timeCheck("backproperrors(): done weight backprop, layer " + toString( layerIndex ) );
        if( !implicitlyCalcedBiasWeight ) {
            doBiasBackpropGpu( learningRate, resultsWrapper, errors, biasWeightChanges );
        StatefulTimer::instance()->timeCheck("backproperrors(): done biasweight backprop, layer " + toString( layerIndex ) );
        }

        if( errorsForUpstream != 0 ) {
            calcErrorsForUpstreamGpu( weightsWrapper, errors, errorsForUpstream );
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
    }

    void updateWeightsGpu( CLWrapper* weightChangesWrapper, CLWrapper*weightsWrapper ) {
        int globalSize = getWeightsSize();
        int workgroupsize = std::min( getWeightsSize(), cl->getMaxWorkgroupSize() );
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
        kernelAddInPlace->in( getWeightsSize() )
            ->in( weightChangesWrapper )
            ->in (weightsWrapper)
            ->run_1d( globalSize, workgroupsize );
        cl->finish();
    }

    void backPropWeightsCpu( float learningRate, float const *errors, float *weights ) {
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

    void backPropWeightsGpu( float learningRate, CLWrapper *imagesWrapper, CLWrapper *resultsWrapper, float const*errors, CLWrapper *weightChangesWrapper ) {
        StatefulTimer::instance()->timeCheck(" backpropweightsGpu start, layer " + toString( layerIndex ) );
        const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
        CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), (float *)errors );
        errorsWrapper->copyToDevice();
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
        delete errorsWrapper;
        StatefulTimer::instance()->timeCheck(" backpropweightsGpu end, layer " + toString( layerIndex ) );
    }

    void backPropWeightsGpuWithScratch( float learningRate, CLWrapper *imagesWrapper, CLWrapper *resultsWrapper, float const*errors, CLWrapper *weightChangesWrapper ) {
//        Timer timer;
        StatefulTimer::instance()->timeCheck(" backpropweightsGpuWithScratch start, layer " + toString( layerIndex ) );
//        int globalSize = getWeightsSize();
        int workgroupsize = filterSizeSquared;
        int numWorkgroups = upstreamNumPlanes * numPlanes;
        int globalSize = workgroupsize * numWorkgroups;
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
//        std::cout << " backpropgpuwithscratch, globalsize " << globalSize << " workgroupsize " << workgroupsize << std::endl;

        const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
        CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), (float *)errors );
//        imagesWrapper->copyToDevice();
        errorsWrapper->copyToDevice();
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
        delete errorsWrapper;
//        delete weightChangesWrapper;
        StatefulTimer::instance()->timeCheck(" backpropweightsGpuWithScratch end, layer " + toString( layerIndex ) );
    }

    void backPropWeightsGpuWithScratchAndBias( float learningRate, CLWrapper *imagesWrapper, CLWrapper *resultsWrapper, float const*errors, CLWrapper *weightChangesWrapper, float *biasWeightChanges ) {
        StatefulTimer::instance()->timeCheck(" backpropweightsGpuWithScratchAndBias start, layer " + toString( layerIndex ) );
        int workgroupsize = filterSizeSquared;
        int numWorkgroups = upstreamNumPlanes * numPlanes;
        int globalSize = workgroupsize * numWorkgroups;
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;

        const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
        CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), (float *)errors );
        CLWrapper *biasWeightChangesWrapper = cl->wrap( numPlanes, biasWeightChanges );
        errorsWrapper->copyToDevice();
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

        delete errorsWrapper;
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
    virtual bool needErrorsBackprop() {
        return true;
    }

    void calcErrorsForUpstreamGpu( CLWrapper *weightsWrapper, float const *const errors, float *const errorsForUpstream ) {
//        CLWrapper *weightsWrapper = cl->wrap( getWeightsSize(), weights );
        StatefulTimer::instance()->timeCheck("calcErrorsForUpstreamGpu start, layer " + toString( layerIndex ) );
        CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), (float *)errors );
        CLWrapper *errorsForUpstreamWrapper = cl->wrap( previousLayer->getResultsSize(), errorsForUpstream );
//        weightsWrapper->copyToDevice();
        errorsWrapper->copyToDevice();
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
        errorsForUpstreamWrapper->copyToHost();

//        StatefulTimer::instance()->timeCheck("    calcErrorsForUpstreamGpu, finished kernel, layer " + toString( layerIndex ) );
//        StatefulTimer::instance()->timeCheck("    calcErrorsForUpstreamGpu, copied results to host, layer " + toString( layerIndex ) );
        delete errorsForUpstreamWrapper;
        delete errorsWrapper;
//        delete weightsWrapper;
        StatefulTimer::instance()->timeCheck("calcErrorsForUpstreamGpu end, layer " + toString( layerIndex ) );
    }

    void calcErrorsForUpstreamCpu( float const *const weights, float const *const errors, float *errorsForUpstream ) {
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

    void doBiasBackpropCpu(float learningRate, float const *results, float const *errors, float *biasWeightChanges ) {
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

    void doBiasBackpropGpu(float learningRate, CLWrapper *resultsWrapper, float const *errors, float *biasWeightChanges ) {
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
        CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), (float *)errors );
        CLWrapper *biasWeightChangesWrapper = cl->wrap( numPlanes, biasWeightChanges );
//        resultsWrapper->copyToDevice();
        errorsWrapper->copyToDevice();

        kernelBackpropBiasWeights->in( learningMultiplier )->in( batchSize )
            ->in( resultsWrapper )->in( errorsWrapper )->out( biasWeightChangesWrapper );
        kernelBackpropBiasWeights->run_1d(globalSize, workgroupsize);
        cl->finish();
        biasWeightChangesWrapper->copyToHost();

        delete biasWeightChangesWrapper;
//        delete resultsWrapper;
        delete errorsWrapper;
        StatefulTimer::instance()->timeCheck("doBiasBackpropGpu end, layer " + toString( layerIndex ) );
    }
};

