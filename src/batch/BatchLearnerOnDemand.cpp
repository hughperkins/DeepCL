// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "normalize/NormalizationHelper.h"
#include "net/NeuralNet.h"
//#include "AccuracyHelper.h"
#include "net/Trainable.h"
#include "loaders/GenericLoader.h"
#include "batch/OnDemandBatcher.h"
#include "batch/BatchLearnerOnDemand.h"
#include "batch/NetAction.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

//BatchLearnerOnDemand::BatchLearnerOnDemand(Trainable *net) :
//    net(net) {
//}

//EpochResult BatchLearnerOnDemand::runBatchedNetAction(std::string filepath, int fileReadBatches, int batchSize, int N, NetAction *netAction) {
//    OnDemandBatcher onDemandBatcher(net, netAction, filepath, N, fileReadBatches, batchSize);
//    return onDemandBatcher.run();
//}

//int BatchLearnerOnDemand::test(std::string filepath, int fileReadBatches, int batchSize, int Ntest) {
//    net->setTraining(false);
//    NetAction *action = new NetForwardAction();
//    int numRight = runBatchedNetAction(filepath, fileReadBatches, batchSize, Ntest, action).numRight;
//    delete action;
//    return numRight;
//}

//EpochResult BatchLearnerOnDemand::runEpochFromLabels(Trainer *trainer, TrainingContext *context,
//         std::string filepath, int fileReadBatches, int batchSize, int Ntrain) {
//    net->setTraining(true);
//    NetAction *action = new NetLearnLabeledAction(trainer, context);
//    EpochResult epochResult = runBatchedNetAction(filepath, fileReadBatches, batchSize, Ntrain, action);
//    delete action;
//    return epochResult;
//}


