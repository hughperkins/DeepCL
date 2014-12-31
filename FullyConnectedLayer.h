// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "ActivationFunction.h"

class FullyConnectedLayer : public Layer {
public:
//    int batchSize;
//    const int upstreamNumPlanes;
//    const int upstreamBoardSize;
//    float *biasWeights;
//    const bool biased;

    FullyConnectedLayer( Layer *previousLayer, FullyConnectedMaker const *maker ) :
            Layer( previousLayer, maker ) {
//        int numPreviousPlanes = previousLayer->getNumPlanes();
//        int previousBoardSize = previousLayer->getBoardSize();
//        int numOutputs = getResultsSize();
        int fanIn = ( upstreamNumPlanes * upstreamBoardSize * upstreamBoardSize );
//        int numThisLayerWeights = fanIn * numOutputs;
        weights = new float[ getWeightsSize() ];
        biasWeights = new float[ getBiasWeightsSize() ];
//        results = 0;
        randomizeWeights( fanIn, weights, getWeightsSize() );
        randomizeWeights( fanIn, biasWeights, getBiasWeightsSize() );
    }

//    FullyConnectedLayer( Layer *previousLayer, int numOutputPlanes, int newBoardSize, bool biased, ActivationFunction *activationFunction ) :
////            previousLayer( previousLayer ),
//            Layer( previousLayer, numOutputPlanes, newBoardSize, activationFunction),
//            upstreamNumPlanes( previousLayer->getNumPlanes() ),
//            upstreamBoardSize( previousLayer->getBoardSize() ),
//            biased( biased ) {
//        int numPreviousPlanes = previousLayer->getNumPlanes();
//        int previousBoardSize = previousLayer->getBoardSize();
//        int numOutputs = (numOutputPlanes * newBoardSize * newBoardSize);
//        int fanIn = ( numPreviousPlanes * previousBoardSize * previousBoardSize );
//        int numThisLayerWeights = fanIn * numOutputs;
//        weights = new float[ numThisLayerWeights ];
//        biasWeights = new float[ numOutputs ];
//        results = 0;
//        randomizeWeights( fanIn, weights, numThisLayerWeights );
//        randomizeWeights( fanIn, biasWeights, numOutputs );
//    }
    // weights like [upstreamPlane][upstreamRow][upstreamCol][outputPlane][outputrow][outputcol]
    inline int getWeightIndex( int prevPlane, int prevRow, int prevCol, int outputPlane, int outputRow, int outputCol ) const {
        int index = ( ( ( ( prevPlane ) * upstreamBoardSize
                       + prevRow ) * upstreamBoardSize
                       + prevCol ) * numPlanes
                       + outputPlane * boardSize 
                       + outputRow ) * boardSize
                       + outputCol;
        return index;
    }
    inline float getWeight( int prevPlane, int prevRow, int prevCol, int outputPlane, int outputRow, int outputCol ) const {
        return weights[getWeightIndex( prevPlane, prevRow, prevCol, outputPlane, outputRow, outputCol )];
    }
    inline int getBiasWeightIndex( int outputPlane, int outputRow, int outputCol ) const {
        int index = (outputPlane * boardSize 
                       + outputRow ) * boardSize
                       + outputCol;
        return index;
    }
    inline float getBiasWeight( int outputPlane, int outputRow, int outputCol ) const {
        return biasWeights[getBiasWeightIndex( outputPlane, outputRow, outputCol ) ];
    }
    virtual void print() const {
        std::cout << "FullyConnectedLayer" << std::endl;
        std::cout << "  numoutputneurons " << numPlanes << std::endl;
        printWeights();
        printOutput();
    }
    virtual void printWeights() const {
        std::cout << "  weights: " << std::endl;
        for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
            for( int outRow = 0; outRow < boardSize; outRow++ ) {
                for( int outCol = 0; outCol < boardSize; outCol++ ) {
                    std::cout << "    outPlane " << outPlane << " outrow " << outRow << " outCol " << outCol << std::endl;
                    if( biased ) std::cout << "    bias: " << getBiasWeight( outPlane, outRow, outCol ) << std::endl;
                    for( int inPlane = 0; inPlane < previousLayer->getNumPlanes(); inPlane++ ) {
                        if( previousLayer->getNumPlanes() > 1 ) std::cout << "    inPlane " << inPlane << std::endl;
                        for( int i = 0; i < std::min( 5, previousLayer->getBoardSize() ); i++ ) {
                            std::cout << "      ";
                            for( int j = 0; j < std::min(5, previousLayer->getBoardSize() ); j++ ) {
                               std::cout << getWeight( inPlane, i, j, outPlane, outRow, outCol ) << " ";
                            }
                            if( previousLayer->getBoardSize() > 5 ) {
                               std::cout << " ... " << " ";
                            }
                            std::cout << std::endl;
                        }
                    }
                    if( previousLayer->getBoardSize() > 5 ) {
                       std::cout << "       ... " << std::endl;
                    }
                    std::cout << std::endl;
                }
            }
        }
    }
    virtual void printOutput() const {
        if( results == 0 ) {
             std::cout << "no results yet" << std::endl;
        } else {
            std::cout << "  outputs: " << std::endl;
            for( int n = 0; n < batchSize; n++ ) {
                std::cout << "    n: " << n << std::endl;
                for( int plane = 0; plane < numPlanes; plane++ ) {
                    if( numPlanes > 1 ) std::cout << "      plane " << plane << std::endl;
                    if( boardSize == 1 ) {
                         std::cout << "        " << getResult(n, plane, 0, 0 ) << std::endl;
                    } else {
                        std::cout << "not implemented for this boardsize" << std::endl;
                    }
                }
            }
        }
    }
    void randomizeWeights(int fanIn, float *weights, int numWeights ) {
//        std::cout << "fullyconnectedlayer randomzing weights" << std::endl;
        for( int i = 0; i < numWeights; i++ ) {
            weights[i] = generateWeight( fanIn );
        }
//        print();
    }
    virtual ~FullyConnectedLayer() {
//        if( results != 0 ) {
//            delete[] results;
//        }
//        delete[] weights;
    }
    virtual void setBatchSize( int batchSize ) {
        if( results != 0 ) {
            delete[] results;
        }
        this->batchSize = batchSize;
        results = new float[numPlanes * batchSize];
        weOwnResults = true;
    }
    virtual void propagate() {
        for( int imageId = 0; imageId < batchSize; imageId++ ) {
            for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
                for( int outRow = 0; outRow < boardSize; outRow++ ) {
                    for( int outCol = 0; outCol < boardSize; outCol++ ) {
                        float sum = 0;
                        for( int inPlane = 0; inPlane < upstreamNumPlanes; inPlane++ ) {
                            for( int inrow = 0; inrow < upstreamBoardSize; inrow++ ) {
                                for( int incol = 0; incol < upstreamBoardSize; incol++ ) {
                                    float thisWeight = getWeight( inPlane, inrow, incol, outPlane, outRow, outCol );
                                    float thisPixel = previousLayer->getResult( imageId, inPlane, inrow, incol );
        //std::cout << "weight " << thisWeight << " pixel " << thisPixel << std::endl;
                                    sum += thisWeight * thisPixel;
                                }
                            }
                        }
                        if( biased ) {
                            sum += 0.5 * getBiasWeight( outPlane, outRow, outCol );
                        }
                        int resultIndex = getResultIndex( imageId, outPlane, outRow, outCol );
        //                results[numPlanes * imageId + out] = activationFn( sum );
                        results[resultIndex] = activationFunction->calc( sum );
        //                std::cout << "n " << imageId << " out " << out << " sum " << sum << " after actfn " << results[resultIndex] << std::endl;
                     }
                }
            }
        }
    }
    // results structured like [imageid][outputplane][outputrow][outputcol]
    virtual void backPropExpected( float learningRate, float const *expected ) {
      //  backPropOld( learningRate, expected );
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
    virtual void backPropErrors( float learningRate, float const *errors ) {
        for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
            for( int outRow = 0; outRow < boardSize; outRow++ ) {
                for( int outCol = 0; outCol < boardSize; outCol++ ) {
                    for( int upstreamPlane = 0; upstreamPlane < upstreamNumPlanes; upstreamPlane++ ) {
                        for( int upstreamRow = 0; upstreamRow < upstreamBoardSize; upstreamRow++ ) {
                            for( int upstreamCol = 0; upstreamCol < upstreamBoardSize; upstreamCol++ ) {
                                float thiswchange = 0;
                                for( int n = 0; n < batchSize; n++ ) {
                                    float upstreamResult = previousLayer->getResult( n, upstreamPlane, upstreamRow, upstreamCol );
                                    int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                                    float actualOutput = results[resultIndex];
                                    float activationDerivative = activationFunction->calcDerivative( actualOutput );
                                    float error = errors[resultIndex];
                                    float thisimagethiswchange = upstreamResult * activationDerivative * error;
                                    thiswchange += thisimagethiswchange;

                                }
                                int weightIndex = getWeightIndex( upstreamPlane, upstreamRow, upstreamCol, outPlane, outRow, outCol );
                                weights[ weightIndex ] -= learningRate * thiswchange / batchSize;
                            }
                        }
                    }
                   // handle bias...
                   if( biased ) {
                       float thiswchange = 0;
                       for( int n = 0; n < batchSize; n++ ) {
                            float upstreamResult = 0.5;
                            int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                            float actualOutput = results[resultIndex];
                            float activationDerivative = activationFunction->calcDerivative( actualOutput );
                            float thisimagethiswchange = upstreamResult * errors[resultIndex] * activationDerivative;
                            thiswchange += thisimagethiswchange;
                       }
                       int biasWeightIndex = getBiasWeightIndex(outPlane, outRow, outCol);
                       biasWeights[ biasWeightIndex ] -= learningRate * thiswchange / batchSize;
                   }
                }
            }
        }
        float *errorsForUpstream = new float[batchSize * upstreamNumPlanes * upstreamBoardSize * upstreamBoardSize];
        for( int n = 0; n < batchSize; n++ ) {
            for( int upstreamPlane = 0; upstreamPlane < upstreamNumPlanes; upstreamPlane++ ) {
                for( int upstreamRow = 0; upstreamRow < upstreamBoardSize; upstreamRow++ ) {
                    for( int upstreamCol = 0; upstreamCol < upstreamBoardSize; upstreamCol++ ) {
                        float sumWeightTimesOutError = 0;
                        for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
                            for( int outRow = 0; outRow < boardSize; outRow++ ) {
                                for( int outCol = 0; outCol < boardSize; outCol++ ) {
                                    int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                                    float thisError = errors[resultIndex];
                                    int thisWeightIndex = getWeightIndex( upstreamPlane, upstreamRow, upstreamCol,
                                       outPlane, outRow, outCol );
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
        previousLayer->backPropErrors(learningRate, errorsForUpstream);
        delete[] errorsForUpstream;
    }
};

