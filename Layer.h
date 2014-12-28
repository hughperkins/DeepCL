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

class Layer {
public:
    int batchSize;
//    int batchStart;
//    int batchEnd;

    Layer *previousLayer;
    float *results;
    bool weOwnResults;
    float *weights;
    const int numPlanes;
    const int boardSize;

    const ActivationFunction *activationFunction;

    Layer( Layer *previousLayer, int numPlanes, int boardSize, ActivationFunction *activationFunction ) :
         previousLayer( previousLayer ),
         numPlanes( numPlanes),
         boardSize( boardSize ),
         results(0),
         weights(0),
         activationFunction( activationFunction ),
         weOwnResults(false) {
    }
    virtual ~Layer() {
        if( results != 0 && weOwnResults ) {
//             std::cout << "Layer, deleting results array " << std::endl;
             delete[] results;
        }
        if( weights != 0 ) {
//             std::cout << "Layer, deleting weights array " << std::endl;
            delete[] weights;
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

