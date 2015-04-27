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
        gradInput( 0 ),
        allocatedSize( 0 ) {
}
VIRTUAL CrossEntropyLoss::~CrossEntropyLoss(){
    if( gradInput != 0 ) {
        delete[] gradInput;
    }
}
VIRTUAL std::string CrossEntropyLoss::getClassName() const {
    return "CrossEntropyLoss";
}
VIRTUAL float*CrossEntropyLoss::getGradInput() {
    return gradInput;
}
VIRTUAL int CrossEntropyLoss::getPersistSize() const {
    return 0;
}
//VIRTUAL float*CrossEntropyLoss::getDerivLossBySumForUpstream() {
//    return errors;
//}
VIRTUAL float CrossEntropyLoss::calcLoss( float const *expected ) {
    float loss = 0;
    float *input = previousLayer->getOutput();
//    cout << "CrossEntropyLoss::calcLoss" << endl;
    // this is matrix subtraction, then element-wise square, then aggregation
    int numPlanes = previousLayer->getOutputPlanes();
    int imageSize = previousLayer->getOutputImageSize();
    for( int imageId = 0; imageId < batchSize; imageId++ ) {
        for( int plane = 0; plane < numPlanes; plane++ ) {
            for( int outRow = 0; outRow < imageSize; outRow++ ) {
                for( int outCol = 0; outCol < imageSize; outCol++ ) {
                    int inputOffset = ( ( imageId
                         * numPlanes + plane )
                         * imageSize + outRow )
                         * imageSize + outCol;
                    float expectedOutput = expected[inputOffset];
                    float inputValue = input[inputOffset];
                    cout << " expected=" << expectedOutput << " input=" << inputValue 
                        << " log(input)=" << log(inputValue) << " log(1-input)" << log(1-inputValue) 
                        << endl;
                    float negthisloss = expectedOutput * log( inputValue ) 
                        + ( 1 - expectedOutput ) * log( 1 - inputValue );
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
    if( gradInput != 0 ) {
        delete[] gradInput;
    }
    gradInput = new float[ batchSize * previousLayer->getOutputSize() ];
    this->batchSize = batchSize;
    allocatedSize = batchSize;
}
// just do naively for now, then add sigmoid short-cutting later
VIRTUAL void CrossEntropyLoss::calcGradInput( float const*expectedOutput ) {
//    ActivationFunction const*fn = previousLayer->getActivationFunction();
    int inputSize = previousLayer->getOutputSize();
    float *input = previousLayer->getOutput();
    for( int i = 0; i < inputSize; i++ ) {
        // TODO: check this :-)
        gradInput[i] = ( input[i] - expectedOutput[i] ) / input[i] / ( 1.0f - input[i] );
    }
}

