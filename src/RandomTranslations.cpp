// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "NeuralNet.h"
#include "Layer.h"
#include "RandomTranslations.h"
#include "MyRandom.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

RandomTranslations::RandomTranslations( Layer *previousLayer, RandomTranslationsMaker *maker ) :
        Layer( previousLayer, maker ),
        translateSize( maker->_translateSize ),
        numPlanes ( previousLayer->getOutputPlanes() ),
        inputBoardSize( previousLayer->getOutputBoardSize() ),
        outputBoardSize( previousLayer->getOutputBoardSize() ),
        results(0),
        batchSize(0),
        allocatedSize(0) {
    if( inputBoardSize == 0 ) {
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString( layerIndex ) + ": input board size is 0" );
    }
    if( outputBoardSize == 0 ) {
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString( layerIndex ) + ": output board size is 0" );
    }
    if( previousLayer->needsBackProp() ) {
        throw runtime_error("Error: RandomTranslations layer does not provide backprop currently, so you cannot put it after a layer that needs backprop");
    }
}
VIRTUAL RandomTranslations::~RandomTranslations() {
    if( results != 0 ) {
        delete[] results;
    }
}
VIRTUAL void RandomTranslations::setBatchSize( int batchSize ) {
    if( batchSize <= allocatedSize ) {
        this->batchSize = batchSize;
        return;
    }
    if( results != 0 ) {
        delete[] results;
    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
    results = new float[ getResultsSize() ];
}
VIRTUAL int RandomTranslations::getResultsSize() {
    return batchSize * numPlanes * outputBoardSize * outputBoardSize;
}
VIRTUAL float *RandomTranslations::getResults() {
    return results;
}
VIRTUAL bool RandomTranslations::needsBackProp() {
    return false;
}
VIRTUAL int RandomTranslations::getResultsSize() const {
    return batchSize * numPlanes * outputBoardSize * outputBoardSize;
}
VIRTUAL int RandomTranslations::getOutputBoardSize() const {
    return outputBoardSize;
}
VIRTUAL int RandomTranslations::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL int RandomTranslations::getPersistSize() const {
    return 0;
}
VIRTUAL bool RandomTranslations::providesErrorsForUpstreamWrapper() const {
    return false;
}
VIRTUAL bool RandomTranslations::hasResultsWrapper() const {
    return false;
}
VIRTUAL void RandomTranslations::propagate() {
    float *upstreamResults = previousLayer->getResults();
//    if( !training ) {
////        cout << "testing, no translation" << endl;
//        int linearSize = getResultsSize();
//        memcpy( results, upstreamResults, linearSize * sizeof(float) );
//        return;
//    }
//    cout << "training => translating" << endl;
//    if( padZeros ) {
//        memset( results, 0, sizeof(float) * getResultsSize() );
//    }
    if( !training ) {
        memcpy( results, upstreamResults, sizeof(float) * getResultsSize() );
        return;
    }
    memset( results, 0, sizeof(float) * getResultsSize() );
    for( int n = 0; n < batchSize; n++ ) {
        for( int plane = 0; plane < numPlanes; plane++ ) {
            float *upstreamBoard = upstreamResults + ( n * numPlanes + plane ) * inputBoardSize * inputBoardSize;
            float *outputBoard = results + ( n * numPlanes + plane ) * outputBoardSize * outputBoardSize;
            const int outRowOffset = MyRandom::instance()->uniformInt( - translateSize, translateSize );
            const int outColOffset = MyRandom::instance()->uniformInt( - translateSize, translateSize );
//            const int outStartRow = outRowOffset > 0 ? rowOffset : 0;
//            const int outEndRow = outputBoardSize - 1  + ( outColOffset < 0 ? outColOffset : 0 );
            const int rowCopyLength = outputBoardSize - abs<int>( outColOffset );
            const int outColStart = outColOffset > 0 ? outColOffset : 0;
            const int inColStart = outColOffset > 0 ? 0 : - outColOffset;
            for( int inRow = 0; inRow < inputBoardSize; inRow++ ) {
                const int outRow = inRow + outRowOffset;
                if( outRow < 0 || outRow >= outputBoardSize - 1 ) {
                    continue;
                }
                memcpy( &(outputBoard[ outRow * outputBoardSize + outColStart ]), 
                    &(upstreamBoard[ inRow * inputBoardSize + inColStart ]),
                    rowCopyLength * sizeof(float) );
            }        
        }
    }
}
VIRTUAL std::string RandomTranslations::asString() const {
    return "RandomTranslations{ inputPlanes=" + toString(numPlanes) + " inputBoardSize=" + toString(inputBoardSize) + " translateSize=" + toString( translateSize ) + " }";
}


