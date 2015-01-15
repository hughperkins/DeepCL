#include "StatefulTimer.h"

#include "SoftMaxLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

SoftMaxLayer::SoftMaxLayer(  Layer *previousLayer, SoftMaxMaker const *maker  ) :
    Layer( previousLayer, maker ),
        allocatedSize( 0 ),
        errorsForUpstream( 0 ) {
}
VIRTUAL SoftMaxLayer::~SoftMaxLayer() {
    if( errorsForUpstream != 0 ) {
        delete[] errorsForUpstream;
    }
}
VIRTUAL float *SoftMaxLayer::getResults() {
    return results;
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
    weOwnResults = true;
    allocatedSize = batchSize;
}
VIRTUAL bool SoftMaxLayer::needErrorsBackprop() {
    return true;
}
VIRTUAL float *SoftMaxLayer::getErrorsForUpstream() {
    return errorsForUpstream;
}
VIRTUAL void SoftMaxLayer::propagate() {
    StatefulTimer::timeCheck("start SoftMaxLayer propagate");
    StatefulTimer::timeCheck("end SoftMaxLayer propagate");
}
VIRTUAL void SoftMaxLayer::backPropErrors( float learningRate ) {
    float *errors = nextLayer->getErrorsForUpstream();
    StatefulTimer::timeCheck("start SoftMaxLayer backproperrors");
    StatefulTimer::timeCheck("end SoftMaxLayer backproperrors");
}

