#ifndef NATUREQLEARNER_H
#define NATUREQLEARNER_H

#include "qlearning/QLearner.h"

class NatureQLearner : public QLearner
{
public:
    NatureQLearner(Trainer *trainer, Scenario *scenario, NeuralNet *net);
    ~NatureQLearner();

    void learnFromPast() override;

    void setReplaceTargetnetEpoch(int value) {replaceTargetnetEpoch = value;}

protected:
    int replaceTargetnetEpoch;
    int netTotalNumWeights;
    NeuralNet * targetNet;
};

#endif // NATUREQLEARNER_H
