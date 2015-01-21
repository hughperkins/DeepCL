// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "StatefulTimer.h"

#include "SoftMaxLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

SoftMaxLayer::SoftMaxLayer(  Layer *previousLayer, SoftMaxMaker const *maker  ) :
    LossLayer( previousLayer, maker ),
        allocatedSize( 0 ),
        errorsForUpstream( 0 ),
        results( 0 ),
        perPlane( maker->_perPlane ),
        boardSize( previousLayer->getOutputBoardSize() ),
        numPlanes( previousLayer->getOutputPlanes() ) {
}
VIRTUAL SoftMaxLayer::~SoftMaxLayer() {
    if( errorsForUpstream != 0 ) {
        delete[] errorsForUpstream;
    }
    if( results != 0 ) {
        delete[] results;
    }
}
VIRTUAL float *SoftMaxLayer::getResults() {
    return results;
}
//VIRTUAL bool SoftMaxLayer::needErrorsBackprop() {
//    return true;
//}
VIRTUAL float *SoftMaxLayer::getErrorsForUpstream() {
    return errorsForUpstream;
}
VIRTUAL void SoftMaxLayer::setBatchSize( int batchSize ) {
    this->batchSize = batchSize;
    if( batchSize <= this->allocatedSize ) {
        return;
    }
    if( results != 0 ) {
        delete[] results;
    }
    if( errorsForUpstream != 0 ) {
        delete[] errorsForUpstream;
    }
    results = new float[ getResultsSize() ];
    errorsForUpstream = new float[ previousLayer-> getResultsSize() ];
    allocatedSize = batchSize;
}

// need to calculate multinomial logistic /cross-entropy loss
VIRTUAL float SoftMaxLayer::calcLoss( float const *expectedValues ) {
//    cout << "softmaxlayer::calcloss" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer calcLoss");
    float loss = 0;
    if( perPlane ) {
        // let's just handle per-column for now, to get this working
        throw std::runtime_error("perPlane not implemented yet.  Sit tight :-)  (But please raise an issue to highlight your need)");
    } else {
        // force boardsize of 1 for now
        if( boardSize != 1 ) {
            throw std::runtime_error("perColumn only supported for boardsize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for( int n = 0; n < batchSize; n++ ) {
            int boardOffset = n * numPlanes * boardSize * boardSize;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                loss -= expectedValues[boardOffset + plane] * log( results[boardOffset + plane] );
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcLoss");
    return loss;
}
// calculate partial deriv loss wrt our inputs, in other words, product of
// (multinomial cross-entropy) loss derivative wrt our output, and
// derivative of softmax wrt our inputs
VIRTUAL void SoftMaxLayer::calcErrors( float const *expectedValues ) {
//    cout << "softmaxlayer::calcerrors" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer calcErrors");
    if( perPlane ) {
        // let's just handle per-column for now, to get this working
        throw std::runtime_error("perPlane not implemented yet.  Sit tight :-)  (But please raise an issue to highlight your need)");
    } else {
        // force boardsize of 1 for now
        if( boardSize != 1 ) {
            throw std::runtime_error("perColumn only supported for boardsize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
//        cout << "numplanes " << numPlanes << endl;
        for( int n = 0; n < batchSize; n++ ) {
            int boardOffset = n * numPlanes * boardSize * boardSize;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                int resultIndex = boardOffset + plane;
                errorsForUpstream[resultIndex] = results[resultIndex] - expectedValues[resultIndex];
//                cout << "plane " << plane << " reuslt " << results[resultIndex] << " expected " << expectedValues[resultIndex] << " error " << errorsForUpstream[resultIndex] << endl;
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcErrors");
}
// for propagate, we just need to apply the softmax activation. "just" :-P
VIRTUAL void SoftMaxLayer::propagate() {
//    cout << "softmaxlayer::propagate" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer propagate");
    if( perPlane ) {
        // let's just handle per-column for now, to get this working
        throw std::runtime_error("perPlane not implemented yet.  Sit tight :-)  (But please raise an issue to highlight your need)");
    } else {
        // force boardsize of 1 for now
        if( boardSize != 1 ) {
            throw std::runtime_error("perColumn only supported for boardsize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        float *resultsFromUpstream = previousLayer->getResults(); // just retrieve as host-side array for now
        for( int n = 0; n < batchSize; n++ ) {
            int boardOffset = n * numPlanes * boardSize * boardSize;
            // first get the max
            float maxValue = resultsFromUpstream[boardOffset + 0]; // since we assume boardsize 1, this is correct
            for( int plane = 1; plane < numPlanes; plane++ ) {
                maxValue = std::max( maxValue, resultsFromUpstream[boardOffset + plane] );
            }
            // calculate sum, under this max
            float denominator = 0;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                denominator += exp( resultsFromUpstream[boardOffset + plane] - maxValue );
            }
            // now calc the softmaxes:
            for( int plane = 0; plane < numPlanes; plane++ ) {
                results[boardOffset + plane] = exp( resultsFromUpstream[boardOffset + plane] - maxValue ) / denominator;
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer propagate");
}
// this seems to be handled by calcErrors? So, just to a nop?
// (cos this layer kind of combines loss layer and a 'normal' propagation layer )
// certainly, we dont have any weights to update, and we already handled error
// propagation in 'calcErrors' method above
VIRTUAL void SoftMaxLayer::backPropErrors( float learningRate ) {
//    cout << "softmaxlayer::backproperrors" << endl;
    // nop, do nothing :-)
}
VIRTUAL std::string SoftMaxLayer::asString() const {
    return "SoftMaxLayer{ perPlane=" + toString( perPlane ) + " numPlanes=" + toString( numPlanes )
        + " boardSize=" + toString( boardSize ) + " }";
}

