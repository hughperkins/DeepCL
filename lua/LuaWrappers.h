#include "NetLearner.h"

class NetLearnerFloats : public NetLearner<float> {
public:
    NetLearnerFloats( NeuralNet *net ) : NetLearner<float>(net) {
    }
};

