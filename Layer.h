// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>
#include <string>
#include <iostream>

#include "MyRandom.h"
#include "ActivationFunction.h"
#include "LayerMaker.h"

class Layer {
public:
//    int batchStart;
//    int batchEnd;

    Layer *previousLayer;
    const int numPlanes;
    const int boardSize;
    float *results;
    float *weights;
    float *biasWeights;
    const bool biased;
    ActivationFunction const *const activationFunction;
    const int upstreamBoardSize;
    const int upstreamNumPlanes;
    const int layerIndex;
    bool weOwnResults;

    int batchSize;

    Layer( Layer *previousLayer, LayerMaker const*maker ) :
        previousLayer( previousLayer ),
        numPlanes( maker->getNumPlanes() ),
        boardSize( maker->getBoardSize() ),
        results(0),
        weights(0),
        biasWeights(0),
        biased(maker->_biased),
        activationFunction( maker->_activationFunction ),
        upstreamBoardSize( previousLayer == 0 ? 0 : previousLayer->boardSize ),
        upstreamNumPlanes( previousLayer == 0 ? 0 : previousLayer->numPlanes ),
        layerIndex( previousLayer == 0 ? 0 : previousLayer->layerIndex + 1 ),
        weOwnResults(false) {
    }

//    Layer( Layer *previousLayer, int numPlanes, int boardSize, ActivationFunction *activationFunction ) :
//         previousLayer( previousLayer ),
//         numPlanes( numPlanes),
//         boardSize( boardSize ),
//         results(0),
//         weights(0),
//         upstreamBoardSize( previousLayer == 0 ? 0 : previousLayer->boardSize )
//         layerIndex( previousLayer == 0 ? 0 : previousLayer->layerIndex + 1 ),
//         activationFunction( activationFunction ),
//         weOwnResults(false) {
//    }
    virtual ~Layer() {
        if( results != 0 && weOwnResults ) {
             delete[] results;
        }
        if( weights != 0 ) {
            delete[] weights;
        }
        if( biasWeights != 0 ) {
            delete[] biasWeights;
        }
        if( activationFunction != 0 ) {
            delete activationFunction;
        }
    }
//    inline float activationFn( float value ) {
//        //return 1.7159 * tanh( value );
//        return tanh( value );
//    }
    virtual void setBatchSize( int batchSize ) { // used to set up internal buffers and stuff
        throw std::runtime_error("setBatchsize not implemetned for this layer type");
    }
    inline float const* getResults() const {
        return results;
    };
    // results structured like [imageid][outputplane][outputrow][outputcol]
    inline int getResultIndex( int n, int plane, int row, int col ) const {
        return ( ( ( n * numPlanes ) + plane ) * boardSize + row ) * boardSize + col;
    }
    inline float getResult( int n, int plane, int row, int col ) const {
        return results[getResultIndex( n, plane,row,col)];
    }
    virtual int getResultsSize() const {
//        throw std::runtime_error("getResultsSize not implemented for this layer type");
         return numPlanes * boardSize * boardSize * batchSize;
    }
    inline int getResultsSizePerExample() const {
        return numPlanes * boardSize * boardSize;
    }
    int getNumPlanes() const {
        return numPlanes;
    }
    int getBoardSize() const {
        return boardSize;
    }
    virtual void propagate() {
        throw std::runtime_error("propagate not implemented for this layer type");
    }
    virtual void print() const { 
//        std::cout << "print() not implemented for this layer type" << std:: endl; 
        printWeights();
        if( results != 0 ) {
            printOutput();
        } else {
            std::cout << "No results yet " << std::endl;
        }
    }
    void initWeights( float*weights ) {
        int numWeights = getWeightsSize();
        for( int i = 0; i < numWeights; i++ ) {
            this->weights[i] = weights[i];
        }
    }
    virtual void printWeightsAsCode() const {
        std::cout << "float weights" << layerIndex << "[] = {";
        const int numWeights = getWeightsSize();
        for( int i = 0; i < numWeights; i++ ) {
            std::cout << weights[i];
            if( i < numWeights - 1 ) std::cout << ", ";
            if( i > 0 && i % 20 == 0 ) std::cout << std::endl;
        }
        std::cout << "};" << std::endl;
//        std::cout << netObjectName << "->layers[" << layerIndex << "]->weights[
    }
    virtual void printBiasWeightsAsCode() const {
        std::cout << "float biasweights" << layerIndex << "[] = {";
        const int numBiasWeights = getBiasWeightsSize();
        for( int i = 0; i < numBiasWeights; i++ ) {
            std::cout << biasWeights[i];
            if( i < numBiasWeights - 1 ) std::cout << ", ";
            if( i > 0 && i % 20 == 0 ) std::cout << std::endl;
        }
        std::cout << "};" << std::endl;
//        std::cout << netObjectName << "->layers[" << layerIndex << "]->weights[
    }
    virtual void printWeights() const { 
        std::cout << "printWeights() not implemented for this layer type" << std:: endl; 
    }
    virtual void printOutput() const { 
        std::cout << "printOutpu() not implemented for this layer type" << std:: endl; 
    }
    static inline float generateWeight( int fanin ) {
        float rangesize = sqrt(12.0f / (float)fanin) ;
//        float uniformrand = random() / (float)random.max();     
        float uniformrand = MyRandom::uniform();   
        float result = rangesize * ( uniformrand - 0.5 );
        return result;
    }
    virtual void backPropExpected( float learningRate, float const *expected ) {
        throw std::runtime_error("backPropExpected not implemented for this layertype");
    }
    virtual void backPropErrors( float learningRate, float const *errors ) {
        throw std::runtime_error("backproperrors not implemented for this layertype");
    }
    virtual int getWeightsSize() const {
        throw std::runtime_error("getWeightsSize not implemented for this layertype");
    }
    virtual int getBiasWeightsSize() const {
        throw std::runtime_error("getBiasWeightsSize not implemented for this layertype");
    }
    float calcLoss( float const *expected ) {
        float E = 0;
        // this is matrix subtraction, then element-wise square, then aggregation
        for( int imageId = 0; imageId < batchSize; imageId++ ) {
            for( int plane = 0; plane < numPlanes; plane++ ) {
                for( int outRow = 0; outRow < boardSize; outRow++ ) {
                    for( int outCol = 0; outCol < boardSize; outCol++ ) {
                        int resultOffset = getResultIndex( imageId, plane, outRow, outCol ); //imageId * numPlanes + out;
                        float expectedOutput = expected[resultOffset];
                        float actualOutput = results[resultOffset];
                        float diff = expectedOutput - actualOutput;
                        float squarederror = diff * diff;
                        E += squarederror;
//                        std::cout << " image " << imageId << " outplane " << plane << " i " << outRow << " j " << outCol <<
//                           " expected " << expectedOutput << " actual " << actualOutput << " squarederror " << squarederror
//                            << std::endl;
                    }
                }
            }            
        }
        return E;
     }
};

