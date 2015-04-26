// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "CrossEntropyLoss.h"
#include "LossLayer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

CrossEntropyLoss::CrossEntropyLoss( Layer *previousLayer, CrossEntropyLossMaker *maker ) :
        LossLayer( previousLayer, maker ),
        errors( 0 ),
        allocatedSize( 0 ) {
}
VIRTUAL CrossEntropyLoss::~CrossEntropyLoss(){
    if( errors != 0 ) {
        delete[] errors;
    }
}
VIRTUAL std::string CrossEntropyLoss::getClassName() const {
    return "CrossEntropyLoss";
}
VIRTUAL float*CrossEntropyLoss::getGradInput() {
    return errors;
}
VIRTUAL int CrossEntropyLoss::getPersistSize() const {
    return 0;
}
//VIRTUAL float*CrossEntropyLoss::getDerivLossBySumForUpstream() {
//    return errors;
//}
VIRTUAL float CrossEntropyLoss::calcLoss( float const *expected ) {
    float loss = 0;
    float *results = getResults();
//    cout << "CrossEntropyLoss::calcLoss" << endl;
    // this is matrix subtraction, then element-wise square, then aggregation
    int numPlanes = previousLayer->getOutputPlanes();
    int imageSize = previousLayer->getOutputImageSize();
    for( int imageId = 0; imageId < batchSize; imageId++ ) {
        for( int plane = 0; plane < numPlanes; plane++ ) {
            for( int outRow = 0; outRow < imageSize; outRow++ ) {
                for( int outCol = 0; outCol < imageSize; outCol++ ) {
                    int resultOffset = ( ( imageId
                         * numPlanes + plane )
                         * imageSize + outRow )
                         * imageSize + outCol;
 //                   int resultOffset = getResultIndex( imageId, plane, outRow, outCol ); //imageId * numPlanes + out;
                    float expectedOutput = expected[resultOffset];
                    float actualOutput = results[resultOffset];
                    float negthisloss = expectedOutput * log( actualOutput ) 
                        + ( 1 - expectedOutput ) * log( 1 - actualOutput );
                    loss -= negthisloss;
                }
            }
        }            
    }
    loss *= 0.5f;
//    cout << "loss " << loss << endl;
    return loss;
 }
VIRTUAL void CrossEntropyLoss::setBatchSize( int batchSize ) {
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
VIRTUAL void CrossEntropyLoss::calcErrors( float const*expectedResults ) {
    ActivationFunction const*fn = previousLayer->getActivationFunction();
    int resultsSize = previousLayer->getResultsSize();
    float *results = previousLayer->getResults();
    for( int i = 0; i < resultsSize; i++ ) {
        float result = results[i];
        float partialOutBySum = fn->calcDerivative( result );
        float partialLossByOut = ( result - expectedResults[i] ) / result / ( 1.0f - result );
        errors[i] = partialLossByOut * partialOutBySum;
    }
}

