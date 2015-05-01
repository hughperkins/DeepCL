// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NormalizationHelper.h"
#include "NeuralNet.h"
//#include "AccuracyHelper.h"
#include "Trainable.h"
#include "GenericLoader.h"
#include "BatchLearner.h"
#include "OnDemandBatcher.h"
#include "BatchLearnerOnDemand.h"
#include "NetAction.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

BatchLearnerOnDemand::BatchLearnerOnDemand( Trainable *net ) :
    net( net ) {
}

EpochResult BatchLearnerOnDemand::runBatchedNetAction( std::string filepath, int fileReadBatches, int batchSize, int N, NetAction *netAction ) {
    OnDemandBatcher onDemandBatcher(net, netAction, filepath, N, fileReadBatches, batchSize );
    return onDemandBatcher.run();
}

int BatchLearnerOnDemand::test( std::string filepath, int fileReadBatches, int batchSize, int Ntest ) {
    net->setTraining( false );
    NetAction *action = new NetForwardAction();
    int numRight = runBatchedNetAction( filepath, fileReadBatches, batchSize, Ntest, action ).numRight;
    delete action;
    return numRight;
}

EpochResult BatchLearnerOnDemand::runEpochFromLabels( Trainer *trainer, std::string filepath, int fileReadBatches, int batchSize, int Ntrain ) {
    net->setTraining( true );
    NetAction *action = new NetLearnLabeledAction( trainer );
    EpochResult epochResult = runBatchedNetAction( filepath, fileReadBatches, batchSize, Ntrain, action );
    delete action;
    return epochResult;
}


