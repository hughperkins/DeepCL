// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "NeuralNet.h"
#include "Layer.h"
#include "RandomPatches.h"
#include "MyRandom.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

RandomPatches::RandomPatches( Layer *previousLayer, RandomPatchesMaker const*maker ) :
        Layer( previousLayer, maker ),
        patchSize( maker->_patchSize ),
        padZeros( maker->_padZeros ),
        numPlanes ( previousLayer->getOutputPlanes() ),
        inputBoardSize( previousLayer->getOutputBoardSize() ),
        outputBoardSize( maker->_padZeros ? previousLayer->getOutputBoardSize() : maker->_patchSize ),
        results(0),
        batchSize(0),
        allocatedSize(0) {
    if( inputBoardSize == 0 ) {
        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString( layerIndex ) + ": input board size is 0" );
    }
    if( outputBoardSize == 0 ) {
        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString( layerIndex ) + ": output board size is 0" );
    }
    if( previousLayer->needsBackProp() ) {
        throw runtime_error("Error: RandomPatches layer does not provide backprop currently, so you cannot put it after a layer that needs backprop");
    }
    if( padZeros ) {
        throw runtime_error( "RandomPatches layer, padzeros not supported yet (though please create an issue, on github, if this is something you want)" );
    }
}
VIRTUAL RandomPatches::~RandomPatches() {
    if( results != 0 ) {
        delete[] results;
    }
}
VIRTUAL void RandomPatches::setBatchSize( int batchSize ) {
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
VIRTUAL int RandomPatches::getResultsSize() {
    return batchSize * numPlanes * outputBoardSize * outputBoardSize;
}
VIRTUAL float *RandomPatches::getResults() {
    return results;
}
VIRTUAL bool RandomPatches::needsBackProp() {
    return false;
}
VIRTUAL int RandomPatches::getResultsSize() const {
    return batchSize * numPlanes * outputBoardSize * outputBoardSize;
}
VIRTUAL int RandomPatches::getOutputBoardSize() const {
    return outputBoardSize;
}
VIRTUAL int RandomPatches::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL int RandomPatches::getPersistSize() const {
    return 0;
}
VIRTUAL bool RandomPatches::providesErrorsForUpstreamWrapper() const {
    return false;
}
VIRTUAL bool RandomPatches::hasResultsWrapper() const {
    return false;
}
VIRTUAL void RandomPatches::propagate() {
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
    for( int n = 0; n < batchSize; n++ ) {
        for( int plane = 0; plane < numPlanes; plane++ ) {
            float *upstreamBoard = upstreamResults + ( n * numPlanes + plane ) * inputBoardSize * inputBoardSize;
            float *outputBoard = results + ( n * numPlanes + plane ) * outputBoardSize * outputBoardSize;
            if( padZeros ) {
                throw runtime_error("padzeros not supported currently in randomtranslations layer");
            } else {
                // in this case, the destination is always exactly the same
                // only the source patch location changes
        //        const int rowOffset = MyRandom::instance()->uniformInt( - translateSize, translateSize );
        //        const int colOffset = MyRandom::instance()->uniformInt( - translateSize, translateSize );
                int patchMargin = inputBoardSize - outputBoardSize;
                int patchRow = patchMargin / 2;
                int patchCol = patchMargin / 2;
                if( training ) {
                    patchRow = MyRandom::instance()->uniformInt( 0, patchMargin );
                    patchCol = MyRandom::instance()->uniformInt( 0, patchMargin );
                }
        //        cout << "patch pos " << patchRow << "," << patchCol << endl;
                for( int outRow = 0; outRow < outputBoardSize; outRow++ ) {
                    const int inRow = outRow + patchRow;
                    memcpy( &(outputBoard[ outRow * outputBoardSize ]), 
                        &(upstreamBoard[ inRow * inputBoardSize + patchCol ]),
                        patchSize * sizeof(float) );
                }        
            }
        }
    }
}
VIRTUAL std::string RandomPatches::asString() const {
    return "RandomPatches{ inputPlanes=" + toString(numPlanes) + " inputBoardSize=" + toString(inputBoardSize) + " patchSize=" + toString( patchSize ) + " padZeros=" + toString(padZeros) + " }";
}


