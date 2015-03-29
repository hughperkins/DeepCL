// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NeuralNet.h"
#include "array_helper.h"

#include "QLearner.h"

using namespace std;

QLearner::QLearner( Scenario *scenario, NeuralNet *net ) :
        scenario( scenario ),
        net( net ) {
    lambda = 0.9f;
    maxSamples = 32;
    epsilon = 0.1f;
    learningRate = 0.1f;

    size = scenario->getPerceptionSize();
    planes = scenario->getPerceptionPlanes();
    numActions = scenario->getNumActions();

    perception = new float[ size * size * planes ];
}

QLearner::~QLearner() {
    delete[] perception;
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
    for( int n = 0; n < batchSize; n++ ) {
        int sampleIdx = myrand() % availableSamples;
        Experience *experience = history[sampleIdx];
        experiences[n] = experience;
    }    

    // copy in data 
    float *afters = new float[ batchSize * 2 * size * size ];
    float *befores = new float[ batchSize * 2 * size * size ];
    for( int n = 0; n < batchSize; n++ ) {
        Experience *experience = experiences[n]; 
        arrayCopy( afters + n * 2 * size * size, experience->after, 2 * size * size );
        arrayCopy( befores + n * 2 * size * size, experience->before, 2 * size * size );
    }

    // get next q values, based on forward prop 'afters'
    net->propagate( afters );
    float const *allResults = net->getResults();
    float *bestQ = new float[ batchSize ];
    int *bestAction = new int[ batchSize ];
    for( int n = 0; n < batchSize; n++ ) {
        float const *results = allResults + n * numActions;
        float thisBestQ = results[0];
        int thisBestAction = 0;
        for( int action = 1; action < numActions; action++ ) {
            if( results[action] > thisBestQ ) {
                thisBestQ = results[action];
                thisBestAction = action;
            }
        }
        bestQ[n] = thisBestQ;
        bestAction[n] = thisBestAction;
    }
    // forward prop 'befores', set up expected values, and backprop
    // new q values
    net->propagate( befores );
    allResults = net->getResults();
    float *expectedValues = new float[ numActions * batchSize ];
    arrayCopy( expectedValues, allResults, batchSize * numActions );
    for( int n = 0; n < batchSize; n++ ) {
        Experience *experience = experiences[n]; 
        if( experience->isEndState ) {
            expectedValues[ n * numActions + experience->action ] = experience->reward; 
        } else {
            expectedValues[ n * numActions + experience->action ] = experience->reward + lambda * bestQ[n];        
        }
    }
    // backprop...
    net->backProp( learningRate / batchSize, expectedValues );
    net->setBatchSize(1);

    delete[] expectedValues;
    delete[] bestQ;
    delete[] bestAction;
    delete[] afters;
    delete[] befores;
    delete[] experiences;
}

void QLearner::run () {
    net->setBatchSize(1);

//    int lastAction = -1;
//    float lastReward = 0.0f;
    float *lastPerception = new float[ size * size * planes ];
//    float *expectedOutputs = new float[ numActions ];
    int game = 0;
//    Experience *experience = 0;
    scenario->getPerception( perception );
    while( true  ) {
//        cout << "see: " << toString( perception, perceptionSize + numActions ) << endl;
        int action = -1;
//        if( lastAction != -1 ) {  
//            float newQ = highestQ * lambda + lastReward;
//            net->propagate( lastPerception );
//            arrayCopy( expectedOutputs, net->getResults(), numActions );
//            expectedOutputs[ lastAction ] = newQ;
//            net->backProp( learningRate, expectedOutputs );
//        }
        if( (myrand() % 10000 / 10000.0f) <= epsilon ) {
            action = myrand() % numActions;
//            cout << "action, rand: " << action << endl;
        } else {
            net->propagate( perception );
            float highestQ = 0;
            int bestAction = 0;
            float const*results = net->getResults();
            for( int i = 0; i < numActions; i++ ) {
                if( i == 0 || results[i] > highestQ ) {
                    highestQ = results[i];
                    bestAction = i;
                }
            }
            action = bestAction;
//            cout << "action, q: " << action << endl;
        }
        arrayCopy( lastPerception, perception, size * size * planes );
//        printDirections( net, scenario->height, scenario->width );
        float reward = scenario->act( action );
        Experience *experience = new Experience(); 
        experience->action = action;
        experience->reward = reward;
        experience->isEndState = scenario->hasFinished();
        experience->before = new float[ size * size * 2 ];
        arrayCopy( experience->before, perception, size * size * 2 );
        scenario->getPerception( perception );
        experience->after = new float[ size * size * 2 ];
        arrayCopy( experience->after, perception, size * size * 2 );
        history.push_back( experience );
//        lastAction = action;
//         lastReward = reward;
//        cout << "reward: " << reward << " lastreward " << lastReward << endl;
//        scenario->print();
        if( scenario->hasFinished() ) {
//            float rewardToEnd = reward;
//            net->propagate( lastPerception );
//            arrayCopy( expectedOutputs, net->getResults(), numActions );
//            expectedOutputs[ lastAction ] = rewardToEnd;
//            net->backProp(learningRate, expectedOutputs );
//            if( game % 10 == 0 ) {
                scenario->print();
                cout << "game " << game << endl;
//                printDirections( scenario, net );
                scenario->printQRepresentation(net);
//            }
//            cout << "scenario finished, resetting..." << endl;
            scenario->reset();
//            lastAction = -1;
//            experience->isEndState = true;
            game++;
        }
        learnFromPast();

//        using namespace std::literals;
//        this_thread::sleep_for( 2s );
//        this_thread::sleep_for( chrono::milliseconds(20) );
    }
}

