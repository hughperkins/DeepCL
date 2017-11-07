#ifndef DOUBLEQLEARNER_H
#define DOUBLEQLEARNER_H

#include "NatureQLearner.h"

class DoubleQLearner : public NatureQLearner
{
public:
    DoubleQLearner(Trainer *trainer, Scenario *scenario, NeuralNet *net);

    void learnFromPast() override;
};

#endif // DOUBLEQLEARNER_H
