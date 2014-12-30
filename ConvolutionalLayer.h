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

class ConvolutionalLayer : public Layer {
public:
    OpenCLHelper *cl;
    CLKernel *kernelConvolve;
//    CLKernel *kernelApplyBias;
//    CLKernel *kernelActivation;
    CLKernel *kernelBackPropWeights;
    CLKernel *kernelBackpropErrors;

    const int filterSize;
    const bool padZeros;
    const int upstreamBoardSize;
    const int upstreamNumPlanes;
    float *biasWeights;
    const bool biased;

    ConvolutionalLayer( Layer *previousLayer, int numFilters, int filterSize, bool padZeros, bool biased, 
            ActivationFunction *activationFunction ) :
                Layer( previousLayer, numFilters, 
                    padZeros ? previousLayer->getBoardSize() : previousLayer->getBoardSize() - filterSize + 1,
                    activationFunction ),
                filterSize( filterSize ),
                padZeros( padZeros ),
                upstreamBoardSize( previousLayer->getBoardSize() ),
                upstreamNumPlanes( previousLayer->getNumPlanes() ),
                biased(biased) {
        std::cout << "convolutionallayer v0.2" << std::endl;
        if( filterSize % 2 == 0 ) {
            throw std::runtime_error("filter size must be an odd number");
        }
        this->cl = new OpenCLHelper();
        std::string options = "-D " + activationFunction->getDefineName();
        if( biased ) {
             options += " -D BIASED";
        }
        this->kernelConvolve = cl->buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2", options );
        this->kernelBackPropWeights = cl->buildKernel( "ClConvolve.cl", "backprop_floats", options );
        this->kernelBackpropErrors = cl->buildKernel( "ClConvolve.cl", "calcErrorsForUpstream", options );
        biasWeights = new float[numPlanes];
        weights = new float[ previousLayer->getNumPlanes() * numPlanes * filterSize * filterSize ];
        randomizeWeights();
    }
    virtual ~ConvolutionalLayer() {
        delete[] biasWeights;
        delete kernelBackPropWeights;
        delete kernelConvolve;
        delete kernelBackpropErrors;
        delete cl;
    }
// filters are organized like [filterid][plane][row][col]
    void randomizeWeights() {
//        print();
        std::cout << "convolutional layer randomzing weights" << std::endl;
//        int numPreviousPlanes = previousLayer->getNumPlanes();
//        int previousBoardSize = previousLayer->getBoardSize();
//        int fanin = numPreviousPlanes * numThisLayerWeights;
        int fanin = upstreamNumPlanes * filterSize * filterSize;
        int numThisLayerWeights = numPlanes * upstreamNumPlanes * filterSize * filterSize;
        std::cout << " numplanes " << numPlanes << " upstreamNumPlanes " << upstreamNumPlanes << 
            " filtersize " << filterSize << " numthislayerweights " << numThisLayerWeights << std::endl;
        for( int i = 0; i < numThisLayerWeights; i++ ) {
            weights[i] = generateWeight( fanin );
        }
        for( int i = 0; i < numPlanes; i++ ) {
            biasWeights[i] = generateWeight( fanin );
        }
        print();
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
        for( int filter = 0; filter < numPlanes; filter++ ) {
           std::cout << "    filter " << filter << std::endl;
           if( biased ) {
               std::cout << "       bias=" << biasWeights[filter] << std::endl;            
           }
           for( int plane = 0; plane < upstreamNumPlanes; plane++ ) {
               if( upstreamNumPlanes > 1 ) std::cout << "    plane " << plane << std::endl;
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
        }
     }
     virtual void printOutput() const {
        if( results == 0 ) {
            return;
        }
        std::cout << "  outputs: " << std::endl;
// results are organized like [imageid][filterid][row][col]
        for( int n = 0; n < batchSize; n++ ) {
            std::cout << "    n: " << n << std::endl;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                if( numPlanes > 1 ) std::cout << "      plane " << plane << std::endl;
                if( boardSize == 1 ) {
                     std::cout << "        " << getResult(n, plane, 0, 0 ) << std::endl;
                } else {
                    for( int i = 0; i < std::min(5,boardSize); i++ ) {
                        std::cout << "      ";
                        for( int j = 0; j < std::min(5,boardSize); j++ ) {
                            std::cout << getResult( n, plane, i, j ) << " ";
    //results[
    //                            n * numPlanes * boardSize*boardSize +
    //                            plane*boardSize*boardSize +
    //                            i * boardSize +
    //                            j ] << " ";
                        }
                        if( boardSize > 5 ) std::cout << " ... ";
                        std::cout << std::endl;
                    }
                    if( boardSize > 5 ) std::cout << " ... " << std::endl;
                }
            }
        }
    }
    virtual void setBatchSize( int batchSize ) {
//        std::cout << "convolutionallayer setting batchsize " << batchSize << std::endl;
        if( results != 0 ) {
//            std::cout << "deleting results array " << std::endl;
            delete[] results;
        }
        this->batchSize = batchSize;
//        std::cout << "allocating results size " << batchSize * numPlanes * boardSize * boardSize << std::endl;
        results = new float[getResultsSize()];
        weOwnResults = true;
    }
    virtual void propagate() {
//        Timer timer;
//        timer.timeCheck("    propagate, created wrappers");
//        for( int i = 0; i < upstreamWrapper->size(); i++ ) {
//            std::cout << "upstreamWrapper[" << i << "]=" << upstreamWrapper->get(i) << std::endl;
//        }
//        for( int i = 0; i < weightsWrapper->size(); i++ ) {
//            std::cout << "weightsWrapper[" << i << "]=" << weightsWrapper->get(i) << std::endl;
//        }

//        std::cout << "propagate, previous result: " << previousLayer->getResults()[0] << " " << previousLayer->getResults()[1] << " size " << batchSize * numPlanes * boardSize * boardSize << std::endl;
//        std::cout << "propagate, weights: " << weights[0] << " " << " size " << previousLayer->getNumPlanes() * numPlanes * filterSize * filterSize << std::endl;

        CLFloatWrapperConst *upstreamWrapper = cl->wrap( previousLayer->getResultsSize(), previousLayer->getResults() );
        CLFloatWrapper *weightsWrapper = cl->wrap( getWeightsSize(), weights );
        CLFloatWrapper *resultsWrapper = cl->wrap( getResultsSize(), results );

        upstreamWrapper->copyToDevice();
        weightsWrapper->copyToDevice();
//        timer.timeCheck("    propagate, copied to device");

        CLFloatWrapper *biasWeightsWrapper = 0;
        if( biased ) {
            biasWeightsWrapper = cl->wrap( getBiasWeightsSize(), biasWeights );
            biasWeightsWrapper->copyToDevice();
        }

//        resultsWrapper->createOnDevice();
        
        kernelConvolve->in( upstreamNumPlanes )->in( numPlanes )->in( upstreamBoardSize )->in( filterSize )
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
//        timer.timeCheck("    propagate, after run");
//        std::cout << "batchsize " << batchSize << " inplanes " << upstreamNumPlanes << " outplanes " << numPlanes << " boardsize " << boardSize 
//           << " filtersize " << filterSize << " padzeros " << padZeros << " globalSize " << globalSize << std::endl;

//        resultsWrapper->copyToHost();
//        for( int i = 0; i < resultsWrapper->size(); i++ ) {
//            std::cout << "results[" << i << "]=" << results[i] << std::endl;
//        }

        resultsWrapper->copyToHost();
//        timer.timeCheck("    propagate, after copy to host");

        delete upstreamWrapper;
        delete weightsWrapper;
        delete resultsWrapper;
        if( biased ) {
            delete biasWeightsWrapper;
        }
    }
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
    virtual void backPropExpected( float learningRate, float const *expected ) {
//        Timer timer;
        float *errors = new float[ batchSize * numPlanes * boardSize * boardSize ];
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
        backPropErrors( learningRate, errors );
        delete[] errors;
    }

    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    // biasweights: [outPlane]
    //       aggregate over:  [upstreamPlane][filterRow][filterCol][outRow][outCol][n]

    virtual void backPropErrors( float learningRate, float const *errors ) {
        Timer timer;
        float *errorsForUpstream = new float[batchSize * upstreamNumPlanes * upstreamBoardSize * upstreamBoardSize];
        const int numWeights = upstreamNumPlanes * numPlanes * filterSize * filterSize;
        float *weightChanges = new float[ numWeights ];
        float *biasWeightChanges = new float[numPlanes];
//        backPropErrors1( learningRate, errors, weightChanges, biasWeightChanges, errorsForUpstream );
        backPropGpu( learningRate, errors, weightChanges );
        timer.timeCheck("backpropgpu");
//        doWeightsBackProp( learningRate, errors, weightChanges );
        doBiasBackprop( learningRate, errors, biasWeightChanges );
        timer.timeCheck("biasbackprop cpu");
//        calcErrorsForUpstream( errors, errorsForUpstream );
//        timer.timeCheck("calcerrorsforupstream cpu");

//        float *weightChanges2 = new float[ numWeights ];
//        float *biasWeightChanges2 = new float[numPlanes];
//        float *errorsForUpstream2 = new float[batchSize * upstreamNumPlanes * upstreamBoardSize * upstreamBoardSize];
//        timer.timeCheck("allocate arrays");

        calcErrorsForUpstreamGpu( errors, errorsForUpstream );
        timer.timeCheck("calcerrorsforupstream gpu");

//        backPropGpu( learningRate, errors, weightChanges2 );
//        for( int i = 0; i < numWeights; i++ ) {
//             float weight2 = weightChanges2[i];
//             if( abs(weightChanges[i] - weight2) / weight2 > 0.000001f ) {
//                 std::cout << "mismatch weight " << i << " " << weightChanges[i] << " != " << weight2 << std::endl;
//             }
//        }

//        for( int plane = 0; plane < numPlanes; plane++ ) {
//             if( abs( biasWeightChanges[plane] - biasWeightChanges2[plane] ) / biasWeightChanges[plane] > 0.000001 ) {
//                  std::cout << "mismatch weight " << plane << " " << biasWeightChanges[plane] << " != " << biasWeightChanges2[plane] << std::endl;
//             }
//        }

//        for( int i = 0; i < batchSize * upstreamNumPlanes * upstreamBoardSize * upstreamBoardSize; i++ ) {
//             if( abs(errorsForUpstream2[i] - errorsForUpstream2[i]) / errorsForUpstream[i] > 0.000001f ) {
//                 std::cout << "mismatch errorsForUpstream2 " << i << " " << errorsForUpstream[i] << " != " << errorsForUpstream2[i] << std::endl;
//             }
//        }
       
        for( int i = 0; i < numWeights; i++ ) {
             weights[i] += weightChanges[i];
        }
        for( int plane = 0; plane < numPlanes; plane++ ) {
            biasWeights[plane] += biasWeightChanges[plane];
        }
        previousLayer->backPropErrors(learningRate, errorsForUpstream);
        delete[] errorsForUpstream;
    }

    void backPropGpu( float learningRate, float const*errors, float *weightChanges ) {
        Timer timer;
        //void kernel backprop_floats_relu( 
        //        const int batchSize, const int upstreamNumPlanes, const int numPlanes, 
        //         const int filterSize, const int outBoardSize, const int padZeros, 
        //         global const float *images, global const float *errors, global float *weightChanges ) {

        const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
        CLWrapper *imagesWrapper = cl->wrap( previousLayer->getResultsSize(), previousLayer->getResults() );
        CLWrapper *resultsWrapper = cl->wrap( getResultsSize(), results );
        CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), errors );
        CLWrapper *weightChangesWrapper = cl->wrap( getWeightsSize(), weightChanges );
        imagesWrapper->copyToDevice();
        resultsWrapper->copyToDevice();
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
        kernelBackPropWeights->run_1d(globalSize, workgroupsize);
        weightChangesWrapper->copyToHost();
        timer.timeCheck("backPropGpu");
    }

    void calcErrorsForUpstreamGpu(  float const *errors, float *errorsForUpstream  ) {
        CLWrapper *weightsWrapper = cl->wrap( getWeightsSize(), weights );
        CLWrapper *errorsWrapper = cl->wrap( getResultsSize(), errors );
        CLWrapper *errorsForUpstreamWrapper = cl->wrap( previousLayer->getResultsSize(), errorsForUpstream );
        weightsWrapper->copyToDevice();
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
        kernelBackpropErrors->run_1d(globalSize, workgroupsize);

        errorsForUpstreamWrapper->copyToHost();
        delete errorsForUpstreamWrapper;
        delete errorsWrapper;
        delete weightsWrapper;
    }

    void calcErrorsForUpstream( float const *errors, float *errorsForUpstream ) {
        Timer timer;
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
                    for( int upstreamCol = 0; upstreamCol < upstreamBoardSize; upstreamCol++ ) {
                        float sumWeightTimesOutError = 0;
                        // aggregate over [outPlane][outRow][outCol]
                        for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
                            for( int outRow = 0; outRow < boardSize; outRow++ ) {
                                // need to derive filterRow and filterCol, given outRow and outCol
                                int filterRow = upstreamRow + margin - outRow;
                                for( int outCol = 0; outCol < boardSize; outCol++ ) {
                                   // need to derive filterRow and filterCol, given outRow and outCol
                                    int filterCol = upstreamCol + margin - outCol;
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
        timer.timeCheck("calced errors for upstream");   
    }

    void doWeightsBackProp( float learningRate, float const *errors, float *weightChanges ) {
        Timer timer;
        const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
        const bool debug = false;
        const int halfFilterSize = filterSize >> 1;
        const int margin = padZeros ? halfFilterSize : 0;
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
    if(debug)std::cout << "outPlane=" << outPlane << " inPlane=" << upstreamPlane << " filterpos=" << filterRow << "," << filterCol
       << " outpos=" << outRow << "," << outCol << " n=" << n << " resindex " << resultIndex << " error=" << error
       << " actualoutput=" << actualOutput << " upstreampos=" << upstreamRow <<"," << upstreamCol << " upstreamResult=" << upstreamResult << " thisimagethiswchange="
       << thisimagethiswchange << std::endl;
                                }
                            }
                        }
//                        weights[ weightIndex ] -= learningRate * thiswchange / batchSize / sqrt( boardSize * boardSize );
                        weightChanges[ weightIndex ] = - thiswchange * learningMultiplier;
                    }
                }
            }
        }
        timer.timeCheck("did backprop to ourselves v2");
    }

    void doBiasBackprop(float learningRate, float const *errors, float *biasWeightChanges ) {
        Timer timer;
        const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
        const bool debug = false;
        if( !biased ) {
             return;
         }
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
        timer.timeCheck("did bias backprop");   
    }

    virtual void backPropErrors1( float learningRate, float const *errors, float *weightChanges, 
             float *biasWeightChanges, float *errorsForUpstream ) {
        Timer timer;
        const bool debug = false;
        const int halfFilterSize = filterSize >> 1;
        const int margin = padZeros ? halfFilterSize : 0;
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
    if(debug)std::cout << "outPlane=" << outPlane << " inPlane=" << upstreamPlane << " filterpos=" << filterRow << "," << filterCol
       << " outpos=" << outRow << "," << outCol << " n=" << n << " resindex " << resultIndex << " error=" << error
       << " actualoutput=" << actualOutput << " upstreampos=" << upstreamRow <<"," << upstreamCol << " upstreamResult=" << upstreamResult << " thisimagethiswchange="
       << thisimagethiswchange << std::endl;
                                }
                            }
                        }
//                        weights[ weightIndex ] -= learningRate * thiswchange / batchSize / sqrt( boardSize * boardSize );
                        weightChanges[ weightIndex ] = - learningRate * thiswchange / batchSize / sqrt( boardSize * boardSize );
                    }
                }
            }
        }
        timer.timeCheck("did backprop to ourselves");
        if( biased ) {
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
//                biasWeights[ outPlane ] -= learningRate * thiswchange / batchSize / sqrt( boardSize * boardSize );
                biasWeightChanges[ outPlane ] = - learningRate * thiswchange / batchSize / sqrt( boardSize * boardSize );
             }
            timer.timeCheck("did bias backprop");   
         }

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
                    for( int upstreamCol = 0; upstreamCol < upstreamBoardSize; upstreamCol++ ) {
                        float sumWeightTimesOutError = 0;
                        // aggregate over [outPlane][outRow][outCol]
                        for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
                            for( int outRow = 0; outRow < boardSize; outRow++ ) {
                                // need to derive filterRow and filterCol, given outRow and outCol
                                int filterRow = upstreamRow + margin - outRow;
                                for( int outCol = 0; outCol < boardSize; outCol++ ) {
                                   // need to derive filterRow and filterCol, given outRow and outCol
                                    int filterCol = upstreamCol + margin - outCol;
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
        timer.timeCheck("calced errors for upstream");   
    }

};

