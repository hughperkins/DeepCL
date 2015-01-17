#include "LossLayer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

LossLayer::LossLayer( Layer *previousLayer, LayerMaker const*maker ) :
        Layer( previousLayer, maker ) {
}
VIRTUAL void LossLayer::propagate() {
}
VIRTUAL float *LossLayer::getResults() {
    return previousLayer->getResults();
}

