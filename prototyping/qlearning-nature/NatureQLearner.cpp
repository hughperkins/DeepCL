#include "net/NeuralNet.h"
#include "qlearning/array_helper.h"
#include "trainers/Trainer.h"
#include "weights/WeightsPersister.h"
#include "NatureQLearner.h"

NatureQLearner::NatureQLearner(Trainer *trainer, Scenario *scenario, NeuralNet *net) : QLearner(trainer, scenario, net)
{
    replaceTargetnetEpoch = 300;
    netTotalNumWeights = WeightsPersister::getTotalNumWeights(net);

    targetNet = net->clone();
}

NatureQLearner::~NatureQLearner()
{
    delete targetNet;
}

void NatureQLearner::learnFromPast()
{
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

    // replace targetNet
    if(epoch % replaceTargetnetEpoch == 0) {
        float weights[netTotalNumWeights];
        WeightsPersister::copyNetWeightsToArray(net, weights);
        WeightsPersister::copyArrayToNetWeights(weights, targetNet);
    }

    // get next q values, based on forward prop 'afters'
    targetNet->setBatchSize(batchSize);
    targetNet->forward(afters);
    float const *allOutput = targetNet->getOutput();
    float *bestQ = new float[ batchSize ];
    for(int n = 0; n < batchSize; n++) {
        float const *output = allOutput + n * numActions;
        float thisBestQ = output[0];
        for(int action = 1; action < numActions; action++) {
            if(output[action] > thisBestQ) {
                thisBestQ = output[action];
            }
        }
        bestQ[n] = thisBestQ;
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
    delete[] afters;
    delete[] befores;
    delete[] experiences;
}
