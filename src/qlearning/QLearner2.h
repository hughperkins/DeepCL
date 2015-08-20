// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// although this looks a bit odd perhaps, this way we dont need to change the existing
// qlearning api much, so dont need to break anything that works already

#pragma once

#include <stdexcept>
#include <iostream>
#include <string>

#include "qlearning/QLearner.h"
#include "trainers/Trainer.h"

class ScenarioProxy : public Scenario {
public:
    const int numActions;
    const int planes;
    const int size;
//    float *perception; // NOT owned by us, dont delete
//    float lastReward;
//    int thisAction;
//    bool isReset;
    ScenarioProxy(int numActions, int planes, int size) :
        numActions(numActions), planes(planes), size(size) {
    }
    virtual int getPerceptionSize() {
        return size;
    }
    virtual int getPerceptionPlanes() {
        return planes;
    }
    virtual void getPerception(float *perception) {
//        perception = this->perception;
        throw std::runtime_error("getPerception not implemented");
    }
    virtual void reset() {
        // noop?
        throw std::runtime_error("reset not implemented");
    }
    virtual int getNumActions() {
//        std::cout << "numActions: " << numActions << std::endl;
        return numActions;
//        throw runtime_error("getNumActions not implemented");
    }
    virtual float act(int index) {
//        this->thisAction = index;
//        return lastReward;
        throw std::runtime_error("act not implemented");
    }
    virtual bool hasFinished() {
//        return isReset;
        throw std::runtime_error("hasFinished not implemented");
    }
};

// The advantage of this over the original QLearning is that it doesnt do any callbacks
// so it should be super-easy to wrap, using Lua, Python, etc ...
class QLearner2 {
    QLearner *qlearner;
    ScenarioProxy *scenario;

    NeuralNet *net;
//    int planes;
//    int size;
//    int numActions;
public:
    QLearner2(Trainer *trainer, NeuralNet *net, int numActions, int planes, int size) : net(net) {
        scenario = new ScenarioProxy(numActions, planes, size);
        qlearner = new QLearner(trainer, scenario, net);
    }
    ~QLearner2() {
        delete qlearner;
        delete scenario;
    }
//    QLearner2 *setPlanes(int planes) {
//        this->planes = planes;
//        scenario->planes = planes;
//        return this;
//    }
//    QLearner2 *setSize(int size) {
//        this->size = size;
//        scenario->size = size;
//        return this;
//    }
//    QLearner2 *setNumActions(int numActions) {
//        this->numActions = numActions;
//        scenario->numActions = numActions;
//        return this;
//    }
    int step(double lastReward, bool wasReset, float *perception) {
//        scenario->lastReward = lastReward;
//        scenario->isReset = isReset;
//        scenario->perception = currentPerception;
        int action = qlearner->step(lastReward, wasReset, perception);
        return action;
    }
    void setLambda(float lambda) { qlearner->setLambda(lambda); }
    void setMaxSamples(int maxSamples) { qlearner->setMaxSamples(maxSamples); }
    void setEpsilon(float epsilon) { qlearner->setEpsilon(epsilon); }
//    void setLearningRate(float learningRate) { qlearner->setLearningRate(learningRate); }
};

