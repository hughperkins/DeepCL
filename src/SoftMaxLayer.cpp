// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "StatefulTimer.h"

#include "LayerMaker.h"
#include "SoftMaxLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

SoftMaxLayer::SoftMaxLayer( Layer *previousLayer, SoftMaxMaker const *maker ) :
    LossLayer( previousLayer, maker ),
        allocatedSize( 0 ),
        errorsForUpstream( 0 ),
        results( 0 ),
        perPlane( maker->_perPlane ),
        boardSize( previousLayer->getOutputBoardSize() ),
        boardSizeSquared( previousLayer->getOutputBoardSize() * previousLayer->getOutputBoardSize() ),
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
VIRTUAL float SoftMaxLayer::calcLossFromLabels( int const *labels ) {
//    cout << "softmaxlayer::calcloss" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer calcLossfromlabels");
    float loss = 0;
    if( perPlane ) {
        for( int n = 0; n < batchSize; n++ ) {
            for( int plane = 0; plane < numPlanes; plane++ ) {
                int label = labels[n * numPlanes + plane];
                int boardOffset = ( n * numPlanes + plane ) * boardSizeSquared;
                loss += - log( results[ boardOffset + label ] );
            }
        }
    } else {
        // force boardsize of 1 for now
        if( boardSize != 1 ) {
            throw std::runtime_error("perColumn only supported for boardsize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for( int n = 0; n < batchSize; n++ ) {
            int boardOffset = n * numPlanes * boardSizeSquared;
            int label = labels[n];
            loss += - log( results[boardOffset + label] );
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcLossfromlabels");
    return loss;
}
// need to calculate multinomial logistic /cross-entropy loss
VIRTUAL float SoftMaxLayer::calcLoss( float const *expectedValues ) {
//    cout << "softmaxlayer::calcloss" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer calcLoss");
    float loss = 0;
    if( perPlane ) {
        for( int n = 0; n < batchSize; n++ ) {
            for( int plane = 0; plane < numPlanes; plane++ ) {
                int boardOffset = ( n * numPlanes + plane ) * boardSizeSquared;
                for( int i = 0; i < boardSizeSquared; i++ ) {
                    if( expectedValues[ boardOffset + i ] != 0 ) {
                        float thisloss = - expectedValues[ boardOffset + i ] * log( results[ boardOffset + i ] );
                        loss += thisloss;
                    }
                }
            }
        }
    } else {
        // force boardsize of 1 for now
        if( boardSize != 1 ) {
            throw std::runtime_error("perColumn only supported for boardsize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for( int n = 0; n < batchSize; n++ ) {
            int boardOffset = n * numPlanes * boardSizeSquared;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                float thisloss = - expectedValues[boardOffset + plane] * log( results[boardOffset + plane] );
//                cout << "n " << n << " plane " << plane << " expected " << expectedValues[boardOffset + plane] << " result " << results[boardOffset + plane] << " thisloss " << thisloss << endl;
                loss += thisloss;
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcLoss");
    return loss;
}
// calculate partial deriv loss wrt our inputs, in other words, product of
// (multinomial cross-entropy) loss derivative wrt our output, and
// derivative of softmax wrt our inputs
VIRTUAL void SoftMaxLayer::calcErrorsFromLabels( int const *labels ) {
//    cout << "softmaxlayer::calcerrors" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer calcErrorsfromlabels");
    if( perPlane ) {
        for( int n = 0; n < batchSize; n++ ) {
            for( int plane = 0; plane < numPlanes; plane++ ) {
                int boardOffset = ( n * numPlanes + plane ) * boardSizeSquared;
                int label = labels[n * numPlanes + plane];
                for( int i = 0; i < boardSizeSquared; i++ ) {
                    errorsForUpstream[boardOffset + i] = results[boardOffset + i];
                }
                errorsForUpstream[boardOffset + label] -= 1;
            }
        }
    } else {
        // force boardsize of 1 for now
        if( boardSize != 1 ) {
            throw std::runtime_error("perColumn only supported for boardsize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for( int n = 0; n < batchSize; n++ ) {
            int boardOffset = n * numPlanes * boardSizeSquared;
            int label = labels[n];
            for( int plane = 0; plane < numPlanes; plane++ ) {
                errorsForUpstream[boardOffset + plane] = results[boardOffset + plane];
            }
            errorsForUpstream[boardOffset + label] -= 1;
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcErrorsfromlabels");
}
// calculate partial deriv loss wrt our inputs, in other words, product of
// (multinomial cross-entropy) loss derivative wrt our output, and
// derivative of softmax wrt our inputs
VIRTUAL void SoftMaxLayer::calcErrors( float const *expectedValues ) {
//    cout << "softmaxlayer::calcerrors" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer calcErrors");
    if( perPlane ) {
        for( int n = 0; n < batchSize; n++ ) {
            for( int plane = 0; plane < numPlanes; plane++ ) {
                int boardOffset = ( n * numPlanes + plane ) * boardSizeSquared;
                for( int i = 0; i < boardSizeSquared; i++ ) {
                    int resultIndex = boardOffset + i;
                    errorsForUpstream[resultIndex] = results[resultIndex] - expectedValues[resultIndex];
                }
            }
        }
    } else {
        // force boardsize of 1 for now
        if( boardSize != 1 ) {
            throw std::runtime_error("perColumn only supported for boardsize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for( int n = 0; n < batchSize; n++ ) {
            int boardOffset = n * numPlanes * boardSizeSquared;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                int resultIndex = boardOffset + plane;
                errorsForUpstream[resultIndex] = results[resultIndex] - expectedValues[resultIndex];
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcErrors");
}
VIRTUAL int SoftMaxLayer::getNumLabelsPerExample() {
    if( perPlane ) {
        return numPlanes;
    } else {
        return boardSizeSquared;
    }
}
VIRTUAL int SoftMaxLayer::getPersistSize() const {
    return 0;
}
VIRTUAL int SoftMaxLayer::calcNumRight( int const*labels ) {
    StatefulTimer::timeCheck("start SoftMaxLayer calcNumRight");
    float *resultsFromUpstream = previousLayer->getResults(); // just retrieve as host-side array for now
    int numRight = 0;
    if( perPlane ) {
        for( int n = 0; n < batchSize; n++ ) {
            for( int plane = 0; plane < numPlanes; plane++ ) {
                int boardOffset = ( n * numPlanes + plane ) * boardSizeSquared;
                int label = labels[n * numPlanes + plane];
                float thisMax = results[boardOffset + 0];
                int iMax = 0;
                for( int i = 1; i < boardSizeSquared; i++ ) {
                    if( results[boardOffset + i] > thisMax ) {
                        thisMax = results[boardOffset + i];
                        iMax = i;
                    }
                }
                if( label == iMax ) {
//                    cout << "n " << n << " plane " << plane << " label " << label << endl;
                    numRight++;
                }
            }
        }
    } else {
        // force boardsize of 1 for now
        if( boardSize != 1 ) {
            throw std::runtime_error("perColumn only supported for boardsize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for( int n = 0; n < batchSize; n++ ) {
            int boardOffset = n * numPlanes * boardSizeSquared;
            int label = labels[n];
            float thisMax = results[boardOffset + 0];
            int iMax = 0;
            for( int i = 1; i < numPlanes; i++ ) {
                if( results[boardOffset + i] > thisMax ) {
                    thisMax = results[boardOffset + i];
                    iMax = i;
                }
            }
            if( label == iMax ) {
                numRight++;
            }
        }
    }

    StatefulTimer::timeCheck("start SoftMaxLayer calcNumRight");
    return numRight;
}
// for propagate, we just need to apply the softmax activation. "just" :-P
VIRTUAL void SoftMaxLayer::propagate() {
//    cout << "softmaxlayer::propagate" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer propagate");
    float *resultsFromUpstream = previousLayer->getResults(); // just retrieve as host-side array for now
    if( perPlane ) {
        for( int n = 0; n < batchSize; n++ ) {
            for( int plane = 0; plane < numPlanes; plane++ ) {
                int boardOffset = ( n * numPlanes + plane ) * boardSizeSquared;
                float maxValue = resultsFromUpstream[boardOffset + 0];
                for( int i = 1; i < boardSizeSquared; i++ ) {
                    maxValue = std::max( maxValue, resultsFromUpstream[boardOffset + i] );
                }
                float denominator = 0;
                for( int i = 0; i < boardSizeSquared; i++ ) {
                    denominator += exp( resultsFromUpstream[boardOffset + i] - maxValue );
                }
                for( int i = 0; i < boardSizeSquared; i++ ) {
                    results[boardOffset + i] = exp( resultsFromUpstream[boardOffset + i] - maxValue ) / denominator;
                }
            }
        }
    } else {
        // force boardsize of 1 for now
        if( boardSize != 1 ) {
            throw std::runtime_error("perColumn only supported for boardsize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for( int n = 0; n < batchSize; n++ ) {
            int boardOffset = n * numPlanes * boardSizeSquared;
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

