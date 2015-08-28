// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "net/Trainable.h"
#include "NetAction.h"
#include "trainers/Trainer.h"
#include "trainers/TrainingContext.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

void NetLearnLabeledAction::run(Trainable *net, int epoch, int batch, float const*const batchData, int const*const batchLabels) {
//    cout << "NetLearnLabeledBatch learningrate=" << learningRate << endl;
    TrainingContext context(epoch, batch);
    trainer->trainFromLabels(net, &context, batchData, batchLabels);
}

void NetForwardAction::run(Trainable *net, int epoch, int batch, float const*const batchData, int const*const batchLabels) {
//    cout << "NetForwardBatch" << endl;
    net->forward(batchData);
//    trainer->train(net, batchData, batchLabels);
}

//void NetBackpropAction::run(Trainable *net, float const*const batchData, int const*const batchLabels) {
////    cout << "NetBackpropBatch learningrate=" << learningRate << endl;
//    net->backwardFromLabels(learningRate, batchLabels);
//}


