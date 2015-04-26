// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "NeuralNet.h"
#include "Layer.h"
#include "RandomTranslations.h"
#include "RandomSingleton.h"
#include "Translator.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

RandomTranslations::RandomTranslations( Layer *previousLayer, RandomTranslationsMaker *maker ) :
        Layer( previousLayer, maker ),
        translateSize( maker->_translateSize ),
        numPlanes ( previousLayer->getOutputPlanes() ),
        inputImageSize( previousLayer->getOutputImageSize() ),
        outputImageSize( previousLayer->getOutputImageSize() ),
        results(0),
        batchSize(0),
        allocatedSize(0) {
    if( inputImageSize == 0 ) {
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString( layerIndex ) + ": input image size is 0" );
    }
    if( outputImageSize == 0 ) {
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString( layerIndex ) + ": output image size is 0" );
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
VIRTUAL std::string RandomTranslations::getClassName() const {
    return "RandomTranslations";
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
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL float *RandomTranslations::getResults() {
    return results;
}
VIRTUAL bool RandomTranslations::needsBackProp() {
    return false;
}
VIRTUAL int RandomTranslations::getResultsSize() const {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL int RandomTranslations::getOutputImageSize() const {
    return outputImageSize;
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
    if( !training ) {
        memcpy( results, upstreamResults, sizeof(float) * getResultsSize() );
        return;
    }
    for( int n = 0; n < batchSize; n++ ) {
        const int translateRows = RandomSingleton::instance()->uniformInt( - translateSize, translateSize );
        const int translateCols = RandomSingleton::instance()->uniformInt( - translateSize, translateSize );
        Translator::translate( n, numPlanes, inputImageSize, translateRows, translateCols, upstreamResults, results );
    }
}
VIRTUAL std::string RandomTranslations::asString() const {
    return "RandomTranslations{ inputPlanes=" + toString(numPlanes) + " inputImageSize=" + toString(inputImageSize) + " translateSize=" + toString( translateSize ) + " }";
}


