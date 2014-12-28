// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "OpenCLHelper.h"
//#include "ClConvolve2.h"

class ConvolutionalLayer : public Layer {
public:
    OpenCLHelper *cl;
    CLKernel *kernelConvolve;
    CLKernel *kernelByElementAddInplace;
    CLKernel *kernelTanh;
    const int filterSize;
    const bool padZeros;
    const int upstreamBoardSize;
    const int upstreamNumPlanes;
    float *biasWeights;
    const bool biased;

    ConvolutionalLayer( Layer *previousLayer, int numFilters, int filterSize, bool padZeros = true, bool biased = true ) :
                Layer( previousLayer, numFilters, 
                    padZeros ? previousLayer->getBoardSize() : previousLayer->getBoardSize() - filterSize + 1 ),
                filterSize( filterSize ),
                padZeros( padZeros ),
                upstreamBoardSize( previousLayer->getBoardSize() ),
                upstreamNumPlanes( previousLayer->getNumPlanes() ),
                biased(biased) {
            if( filterSize % 2 == 0 ) {
                throw std::runtime_error("filter size must be an odd number");
            }
            this->cl = new OpenCLHelper();
//        if( padZeros ) {
            this->kernelConvolve = cl->buildKernel( "ClConvolve.cl", "convolve_imagecubes_float2" );
            this->kernelByElementAddInplace = cl->buildKernel( "ClConvolve.cl", "byelement_add_inplace" );
            this->kernelTanh = cl->buildKernel( "ClConvolve.cl", "byelement_tanh" );
//        } else {
//            this->kernel = cl->buildKernel( "ClConvolve.cl", "convolve_imagecubes_float_nopadzeros" );
//        }
        biasWeights = new float[numPlanes];
        weights = new float[ previousLayer->getNumPlanes() * numPlanes * filterSize * filterSize ];
        randomizeWeights();
    }
    virtual ~ConvolutionalLayer() {
        delete[] biasWeights;
        delete kernelByElementAddInplace;
        delete kernelConvolve;
        delete kernelTanh;
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
    virtual void print() {
        std::cout << "ConvolutionalLayer numFilters " << numPlanes << " filtersize " << filterSize << 
            " padZeros " << padZeros << " biased " << biased << " outputBoardSize " << boardSize << std::endl;
        printWeights();
        if( results != 0 ) {
            printOutputs();
        }
    }
    virtual void printWeights() {
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
     virtual void printOutputs() {
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
        results = new float[batchSize * numPlanes * boardSize * boardSize];
        weOwnResults = true;
    }
    virtual void propagate() {

        CLFloatWrapperConst *upstreamWrapper = cl->wrap( batchSize * numPlanes * upstreamBoardSize * upstreamBoardSize, previousLayer->getResults() );
        CLFloatWrapper *weightsWrapper = cl->wrap( upstreamNumPlanes * numPlanes * filterSize * filterSize, 
                 weights );
        CLFloatWrapper *resultsWrapper = cl->wrap( batchSize * numPlanes * boardSize * boardSize, results );

//        for( int i = 0; i < upstreamWrapper->size(); i++ ) {
//            std::cout << "upstreamWrapper[" << i << "]=" << upstreamWrapper->get(i) << std::endl;
//        }
//        for( int i = 0; i < weightsWrapper->size(); i++ ) {
//            std::cout << "weightsWrapper[" << i << "]=" << weightsWrapper->get(i) << std::endl;
//        }

//        std::cout << "propagate, previous result: " << previousLayer->getResults()[0] << " " << previousLayer->getResults()[1] << " size " << batchSize * numPlanes * boardSize * boardSize << std::endl;
//        std::cout << "propagate, weights: " << weights[0] << " " << " size " << previousLayer->getNumPlanes() * numPlanes * filterSize * filterSize << std::endl;

        upstreamWrapper->copyToDevice();
        weightsWrapper->copyToDevice();

//        resultsWrapper->createOnDevice();
        
        kernelConvolve->in( upstreamNumPlanes )->in( numPlanes )->in( upstreamBoardSize )->in( filterSize )
          ->in( padZeros ? 1 : 0 );
        kernelConvolve->input( upstreamWrapper );
        kernelConvolve->input( weightsWrapper);
        kernelConvolve->output( resultsWrapper );
        int globalSize = batchSize * numPlanes * boardSize * boardSize;
        int workgroupsize = cl->getMaxWorkgroupSize();
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
        kernelConvolve->run_1d( globalSize, workgroupsize );
//        std::cout << "batchsize " << batchSize << " inplanes " << upstreamNumPlanes << " outplanes " << numPlanes << " boardsize " << boardSize 
//           << " filtersize " << filterSize << " padzeros " << padZeros << " globalSize " << globalSize << std::endl;

//        resultsWrapper->copyToHost();
//        for( int i = 0; i < resultsWrapper->size(); i++ ) {
//            std::cout << "results[" << i << "]=" << results[i] << std::endl;
//        }

        if( biased ) {
            CLWrapper *biasWrapper = cl->wrap( numPlanes, biasWeights );
            biasWrapper->copyToDevice();
            kernelByElementAddInplace->inout( resultsWrapper)->input( biasWrapper );
            kernelByElementAddInplace->run_1d( globalSize, workgroupsize );
        }

        kernelTanh->inout( resultsWrapper );
        kernelTanh->run_1d( globalSize, workgroupsize );
        resultsWrapper->copyToHost();

        delete upstreamWrapper;
        delete weightsWrapper;
        delete resultsWrapper;
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
        backPropErrors( learningRate, errors );
        delete[] errors;
    }
    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
    //       aggregate over:  [outRow][outCol][n]
    // biasweights: [outPlane]
    //       aggregate over:  [upstreamPlane][filterRow][filterCol][outRow][outCol][n]
    virtual void backPropErrors( float learningRate, float const *errors ) {
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
                                    float activationDerivative = 1 - actualOutput * actualOutput;
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
                        weights[ weightIndex ] -= learningRate * thiswchange / batchSize / sqrt( boardSize * boardSize );
                    }
                }
            }
        }
        if( biased ) {
             for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
                // bias...
                // biasweights: [outPlane]
                //       aggregate over:  [upstreamPlane][filterRow][filterCol][outRow][outCol][n]
                float thiswchange = 0;
                for( int n = 0; n < batchSize; n++ ) {
                    for( int outRow = 0; outRow < boardSize; outRow++ ) {
                        for( int outCol = 0; outCol < boardSize; outCol++ ) {
                            float upstreamResult = 0.5;
                            int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                            float actualOutput = results[resultIndex];
                            float activationDerivative = 1 - actualOutput * actualOutput;
                            float thisimagethiswchange = upstreamResult * errors[resultIndex] * activationDerivative;
                            thiswchange += thisimagethiswchange;
        if(debug)std::cout << "bias outPlane=" << outPlane << " outpos=" << outRow << "," << outCol << " n=" << n << " resindex " << resultIndex << " error=" << errors[resultIndex]
           << " actualoutput=" << actualOutput << " upstreamResult=" << upstreamResult << " thisimagethiswchange="
           << thisimagethiswchange << std::endl;
                        }
                    }
                }
                biasWeights[ outPlane ] -= learningRate * thiswchange / batchSize / sqrt( boardSize * boardSize );
             }
         }
    }
};

