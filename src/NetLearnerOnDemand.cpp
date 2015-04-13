// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "StatefulTimer.h"
#include "Timer.h"
#include "BatchLearnerOnDemand.h"
#include "NeuralNet.h"
#include "Trainable.h"

#include "NetLearnerOnDemand.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

template< typename T > NetLearnerOnDemand<T>::NetLearnerOnDemand( Trainable *net ) :
        net( net ) {
    batchSize = 128;
    annealLearningRate = 1.0f;
    numEpochs = 12;
    startEpoch = 1;
    dumpTimings = false;
//    trainData = 0;
//    trainLabels = 0;
//    testData = 0;
//    testLabels = 0;
}

template< typename T > void NetLearnerOnDemand<T>::setTrainingData( std::string trainFilepath, int Ntrain ) {
    this->Ntrain = Ntrain;
    this->trainFilepath = trainFilepath;
//    this->trainData = trainData;
//    this->trainLabels = trainLabels;
}

template< typename T > void NetLearnerOnDemand<T>::setTestingData( std::string testFilepath, int Ntest ) {
    this->testFilepath = testFilepath;
    this->Ntest = Ntest;
}

template< typename T > void NetLearnerOnDemand<T>::setSchedule( int numEpochs ) {
    setSchedule( numEpochs, 1 );
}

template< typename T > void NetLearnerOnDemand<T>::setDumpTimings( bool dumpTimings ) {
    this->dumpTimings = dumpTimings;
}

template< typename T > void NetLearnerOnDemand<T>::setSchedule( int numEpochs, int startEpoch ) {
    this->numEpochs = numEpochs;
    this->startEpoch = startEpoch;
}

template< typename T > void NetLearnerOnDemand<T>::setBatchSize( int fileReadBatches, int batchSize ) {
    this->batchSize = batchSize;
    this->fileReadBatches = fileReadBatches;
}

template< typename T > VIRTUAL NetLearnerOnDemand<T>::~NetLearnerOnDemand() {
//    for( vector<PostEpochAction *>::iterator it = postEpochActions.begin(); it != postEpochActions.end(); it++ ) {
//        delete (*it);
//    }
}

template< typename T > VIRTUAL void NetLearnerOnDemand<T>::addPostEpochAction( PostEpochAction *action ) {
    postEpochActions.push_back( action );
}
template< typename T > VIRTUAL void NetLearnerOnDemand<T>::addPostBatchAction( NetLearner_PostBatchAction *action ) {
    postBatchActions.push_back( action );
}
template< typename T > void NetLearnerOnDemand<T>::learn( float learningRate ) {
    learn( learningRate, 1.0f );
}

template< typename T > void NetLearnerOnDemand<T>::learn( float learningRate, float annealLearningRate ) {
//    trainData = new T[ batchSize * net->getInputCubeSize() ];
//    trainLabels = new int[ batchSize ];
//    testData = new T[ batchSize * net->getInputCubeSize() ];
//    testLabels = new int[ batchSize ];
    BatchLearnerOnDemand<T> batchLearnerOnDemand( net );
    NetLearnerPostBatchRunner postRunner;
    batchLearnerOnDemand.addPostBatchAction( &postRunner );
    for( vector<NetLearner_PostBatchAction *>::iterator it = postBatchActions.begin(); it != postBatchActions.end(); it++ ) {
        postRunner.postBatchActions.push_back( *it );
    }
    Timer timer;
    for( int epoch = startEpoch; epoch <= numEpochs; epoch++ ) {
        postRunner.epoch = epoch;
        float annealedLearningRate = learningRate * pow( annealLearningRate, epoch );
        EpochResult epochResult = batchLearnerOnDemand.runEpochFromLabels( annealedLearningRate, trainFilepath, fileReadBatches, batchSize, Ntrain );
        cout << "dumpTimings " << dumpTimings << endl;
        if( dumpTimings ) {
            StatefulTimer::dump(true);
        }
//        cout << "-----------------------" << endl;
        cout << endl;
        timer.timeCheck("after epoch " + toString(epoch ) );
        cout << "annealed learning rate: " << annealedLearningRate << " training loss: " << epochResult.loss << endl;
        cout << " train accuracy: " << epochResult.numRight << "/" << Ntrain << " " << (epochResult.numRight * 100.0f/ Ntrain) << "%" << std::endl;
        int testNumRight = batchLearnerOnDemand.test( testFilepath, fileReadBatches, batchSize, Ntest );
        cout << "test accuracy: " << testNumRight << "/" << Ntest << " " << (testNumRight * 100.0f / Ntest ) << "%" << endl;
        timer.timeCheck("after tests");
        for( vector<PostEpochAction *>::iterator it = postEpochActions.begin(); it != postEpochActions.end(); it++ ) {
            (*it)->run( epoch );
        }
    }
}

template class NetLearnerOnDemand<unsigned char>;
//template class NetLearnerOnDemand<float>;

