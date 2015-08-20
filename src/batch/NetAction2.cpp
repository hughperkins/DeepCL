// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "net/Trainable.h"
#include "batch/NetAction2.h"
#include "batch/Batcher2.h"
#include "trainers/Trainer.h"
#include "trainers/TrainingContext.h"
#include "batch/NetAction.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

void NetLearnAction2::run(Trainable *net, int epoch, int batch, InputData *inputData, OutputData *outputData) {
//    cout << "NetLearnLabeledBatch learningrate=" << learningRate << endl;
    TrainingContext context(epoch, batch);
    ExpectedData *expected = dynamic_cast< ExpectedData * >(outputData);
    LabeledData *labeled = dynamic_cast< LabeledData * >(outputData);
    BatchResult batchResult;
    if(expected != 0) {
        batchResult = trainer->train(net, &context, inputData->inputs, expected->expected);
    } else if(labeled != 0) {
        batchResult = trainer->trainFromLabels(net, &context, inputData->inputs, labeled->labels);        
    }
    epochLoss += batchResult.loss;
    epochNumRight += batchResult.numRight;
}

void NetForwardAction2::run(Trainable *net, int epoch, int batch, InputData *inputData, OutputData *outputData) {
//    cout << "NetForwardBatch" << endl;
    net->forward(inputData->inputs);
//    trainer->train(net, batchData, batchLabels);
}

//void NetBackpropAction::run(Trainable *net, InputData *inputData, OutputData *outputData) {
////    cout << "NetBackpropBatch learningrate=" << learningRate << endl;
//    net->backwardFromLabels(learningRate, batchLabels);
//}


