// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "net/NeuralNet.h"
#include "qlearning/array_helper.h"
#include "trainers/Trainer.h"
#include "qlearning/QLearner.h"

using namespace std;

QLearner::QLearner(Trainer *trainer, Scenario *scenario, NeuralNet *net) :
        trainer(trainer),
        scenario(scenario),
        net(net) {
    epoch = 0;
    lambda = 0.9f;
    maxSamples = 32;
    epsilon = 0.1f;
//    learningRate = 0.1f;

    size = scenario->getPerceptionSize();
    planes = scenario->getPerceptionPlanes();
    numActions = scenario->getNumActions();

    lastPerception = new float[ size * size * planes ];
    game = 0;
    lastAction = -1;
}

QLearner::~QLearner() {
    delete[] lastPerception;
}

void QLearner::learnFromPast() {
    const int availableSamples = history.size();
    int batchSize = availableSamples >= maxSamples ? maxSamples : availableSamples;
//    batchSize = ;
    const int size = scenario->getPerceptionSize();
    const int numActions = scenario->getNumActions();
    net->setBatchSize(batchSize);

//    cout << "batchSize: " << batchSize << endl;

    // draw samples
    Experience **experiences = new Experience *[ batchSize ];
    for(int n = 0; n < batchSize; n++) {
        int sampleIdx = myrand() % availableSamples;
        Experience *experience = history[sampleIdx];
        experiences[n] = experience;
    }    

    // copy in data 
    float *afters = new float[ batchSize * planes * size * size ];
    float *befores = new float[ batchSize * planes * size * size ];
    for(int n = 0; n < batchSize; n++) {
        Experience *experience = experiences[n]; 
        arrayCopy(afters + n * planes * size * size, experience->after, planes * size * size);
        arrayCopy(befores + n * planes * size * size, experience->before, planes * size * size);
    }

    // get next q values, based on forward prop 'afters'
    net->forward(afters);
    float const *allOutput = net->getOutput();
    float *bestQ = new float[ batchSize ];
    int *bestAction = new int[ batchSize ];
    for(int n = 0; n < batchSize; n++) {
        float const *output = allOutput + n * numActions;
        float thisBestQ = output[0];
        int thisBestAction = 0;
        for(int action = 1; action < numActions; action++) {
            if(output[action] > thisBestQ) {
                thisBestQ = output[action];
                thisBestAction = action;
            }
        }
        bestQ[n] = thisBestQ;
        bestAction[n] = thisBestAction;
    }
    // forward prop 'befores', set up expected values, and backprop
    // new q values
    net->forward(befores);
    allOutput = net->getOutput();
    float *expectedValues = new float[ numActions * batchSize ];
    arrayCopy(expectedValues, allOutput, batchSize * numActions);
    for(int n = 0; n < batchSize; n++) {
        Experience *experience = experiences[n]; 
        if(experience->isEndState) {
            expectedValues[ n * numActions + experience->action ] = experience->reward; 
        } else {
            expectedValues[ n * numActions + experience->action ] = experience->reward + lambda * bestQ[n];        
        }
    }
    // backprop...
//    throw runtime_error("need to implement this");
    TrainingContext context(epoch, 0);
    trainer->train(net, &context, befores, expectedValues);
//    net->backward(learningRate / batchSize, expectedValues);
    net->setBatchSize(1);

    epoch++;

    delete[] expectedValues;
    delete[] bestQ;
    delete[] bestAction;
    delete[] afters;
    delete[] befores;
    delete[] experiences;
}

// this is now a scenario-free zone, and therefore no callbacks, and easy to wrap with
// swig, cython etc.
int QLearner::step(float lastReward, bool wasReset, float *perception) { // do one frame
    if(lastAction != -1) {
        Experience *experience = new Experience(); 
        experience->action = lastAction;
        experience->reward = lastReward;
        experience->isEndState = wasReset;
        experience->before = new float[ size * size * planes ];
        arrayCopy(experience->before, this->lastPerception, size * size * planes);
//        scenario->getPerception(perception);
        experience->after = new float[ size * size * planes ];
        arrayCopy(experience->after, perception, size * size * planes);
        history.push_back(experience);
        if(wasReset) {
            game++;
        }
        learnFromPast();
    }
//        cout << "see: " << toString(perception, perceptionSize + numActions) << endl;
    int action = -1;
    if(lastAction == -1 || (myrand() % 10000 / 10000.0f) <= epsilon) {
        action = myrand() % numActions;
//            cout << "action, rand: " << action << endl;
    } else {
        net->setBatchSize(1);
        net->forward(perception);
        float highestQ = 0;
        int bestAction = 0;
        float const*output = net->getOutput();
        for(int i = 0; i < numActions; i++) {
            if(i == 0 || output[i] > highestQ) {
                highestQ = output[i];
                bestAction = i;
            }
        }
        action = bestAction;
//            cout << "action, q: " << action << endl;
    }
    arrayCopy(this->lastPerception, perception, size * size * planes);
//        printDirections(net, scenario->height, scenario->width);
    this->lastAction = action;
    return action;
}

void QLearner::run() {
    game = 0;

    float lastReward = 0;
//    int selectedAction = -1;
    float *perception = new float[ size * size * planes ];
    bool wasReset = false;
    while(true) {
        scenario->getPerception(perception);
        int action = step(lastReward, wasReset, perception);
        lastReward = scenario->act(action);
        if(scenario->hasFinished()) {
            scenario->reset();
            wasReset = true;
        } else {
            wasReset = false;
        }
    }
    delete[] perception; // I guess we will never get to here :-P
}

