// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "MultinomialCrossEntropy.h"
#include "LossLayer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

MultinomialCrossEntropy::MultinomialCrossEntropy( Layer *previousLayer, MultinomialCrossEntropyMaker const*maker ) :
        LossLayer( previousLayer, maker ),
        errors( 0 ),
        allocatedSize( 0 ) {
}
VIRTUAL MultinomialCrossEntropy::~MultinomialCrossEntropy(){
    if( errors != 0 ) {
        delete[] errors;
    }
}
VIRTUAL float*MultinomialCrossEntropy::getErrorsForUpstream() {
    return errors;
}
VIRTUAL float MultinomialCrossEntropy::calcLoss( float const *expected ) {
    float loss = 0;
    float *results = getResults();
//    cout << "MultinomialCrossEntropy::calcLoss" << endl;
    // this is matrix subtraction, then element-wise square, then aggregation
    int numPlanes = previousLayer->getOutputPlanes();
    int boardSize = previousLayer->getOutputBoardSize();
    for( int imageId = 0; imageId < batchSize; imageId++ ) {
        for( int plane = 0; plane < numPlanes; plane++ ) {
            for( int outRow = 0; outRow < boardSize; outRow++ ) {
                for( int outCol = 0; outCol < boardSize; outCol++ ) {
                    int resultOffset = ( ( imageId
                         * numPlanes + plane )
                         * boardSize + outRow )
                         * boardSize + outCol;
 //                   int resultOffset = getResultIndex( imageId, plane, outRow, outCol ); //imageId * numPlanes + out;
                    float expectedOutput = expected[resultOffset];
                    float actualOutput = results[resultOffset];
                    float negthisloss = expectedOutput * log( actualOutput );
                    loss -= negthisloss;
                }
            }
        }            
    }
    loss *= 0.5f;
//    cout << "loss " << loss << endl;
    return loss;
 }
VIRTUAL void MultinomialCrossEntropy::setBatchSize( int batchSize ) {
    if( batchSize <= allocatedSize ) {
        this->batchSize = batchSize;
        return;
    }
    if( errors != 0 ) {
        delete[] errors;
    }
    errors = new float[ batchSize * previousLayer->getResultsSize() ];
    this->batchSize = batchSize;
    allocatedSize = batchSize;
}
// just do naively for now, then add sigmoid short-cutting later
VIRTUAL void MultinomialCrossEntropy::calcErrors( float const*expectedResults ) {
    ActivationFunction const*fn = previousLayer->getActivationFunction();
    int resultsSize = previousLayer->getResultsSize();
    float *results = previousLayer->getResults();
    for( int i = 0; i < resultsSize; i++ ) {
        float result = results[i];
        float partialOutBySum = fn->calcDerivative( result );
        float partialLossByOut = - expectedResults[i] / result;
        errors[i] = partialLossByOut * partialOutBySum;
    }
}

