#pragma once

#include "Layer.h"

class ExpectedValuesLayerMaker;

// holds the expected values for ... a batch? all batches? thinking... should be: one batch
// might be migrated to hold the expected values in gpu buffer later, perhaps?
// or maybe no need?
// note: cant put the expected values in the constructor, since:
//  - this object itself is persistent across batches
//  - but the errors/expected values are in fact: not
class ExpectedValuesLayer : public Layer {
public:
    float *errors;
    int allocatedSize;
    int batchSize;
    ExpectedValuesLayer(Layer *previousLayer, ExpectedValuesLayerMaker const*maker) :
        Layer(previousLayer, maker),
        allocatedSize(0),
        errors(0),
        batchSize(0) {
//        std::cout << "ExpectedValuesLayer()" << std::endl;
    }
    virtual ~ExpectedValuesLayer() {
//        std::cout << "~ExpectedValuesLayer()" << std::endl;
        if( errors != 0 ) {
            delete[] errors;
        }
    }
//    virtual bool needErrorsBackProp() const {
//        return true;
//    }
    virtual bool providesErrorsWrapper() const {
        return false;
    }
    virtual bool needsErrors() const {
        return false;
    }
    virtual float *getErrorsForUpstream() {
        return errors;
    }
    virtual void setBatchSize( int batchSize ) {
        if( batchSize <= allocatedSize ) {
            this->batchSize = batchSize;
            return;
        }
        if( errors != 0 ) {
            delete[] errors;
        }
//        std::cout << "expectedvalueslayer allocating errors, " << previousLayer->getResultsSize() << " floats " << std::endl;
        errors = new float[ previousLayer->getResultsSize() ];
        allocatedSize = batchSize;
        this->batchSize = batchSize;
    }
    void calcErrors( float const *expected ) {
        StatefulTimer::timeCheck("expectedvalueslayer: calcerrors start");
//        getResults();
        if( errors == 0 ) {
            throw std::runtime_error("Need to call setBatchSize on ExpectedValuesLayer");
        }
        float *results = previousLayer->getResults();
//        if( errors == 0 ) {
//            errors = new float[ previousLayer->getResultsSize() ];
//        }
        // matrix per-element subtraction...
        for( int n = 0; n < batchSize; n++ ) {
            for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
                for( int outRow = 0; outRow < boardSize; outRow++ ) {
                    for( int outCol = 0; outCol < boardSize; outCol++ ) {
                        int resultIndex = previousLayer->getResultIndex( n, outPlane, outRow, outCol );
                        errors[ resultIndex ] = results[resultIndex] - expected[resultIndex];
                    }
                } 
            }
        }
        StatefulTimer::timeCheck("expectedvalueslayer: calcerrors end");
    }
};

