// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "StatefulTimer.h"
#include "Timer.h"
#include "BatchLearner.h"
#include "NeuralNet.h"
#include "Trainable.h"

#include "NetLearner.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

template< typename T > NetLearner<T>::NetLearner( Trainable *net ) :
        net( net ) {
    batchSize = 128;
    annealLearningRate = 1.0f;
    numEpochs = 12;
    startEpoch = 1;
    dumpTimings = false;
}

template< typename T > void NetLearner<T>::setTrainingData( int Ntrain, T *trainData, int *trainLabels ) {
    this->Ntrain = Ntrain;
    this->trainData = trainData;
    this->trainLabels = trainLabels;
}

template< typename T > void NetLearner<T>::setTestingData( int Ntest, T *testData, int *testLabels ) {
    this->Ntest = Ntest;
    this->testData = testData;
    this->testLabels = testLabels;
}

template< typename T > void NetLearner<T>::setSchedule( int numEpochs ) {
    setSchedule( numEpochs, 1 );
}

template< typename T > void NetLearner<T>::setDumpTimings( bool dumpTimings ) {
    this->dumpTimings = dumpTimings;
}

template< typename T > void NetLearner<T>::setSchedule( int numEpochs, int startEpoch ) {
    this->numEpochs = numEpochs;
    this->startEpoch = startEpoch;
}

template< typename T > void NetLearner<T>::setBatchSize( int batchSize ) {
    this->batchSize = batchSize;
}

template< typename T > VIRTUAL NetLearner<T>::~NetLearner() {
//    for( vector<PostEpochAction *>::iterator it = postEpochActions.begin(); it != postEpochActions.end(); it++ ) {
//        delete (*it);
//    }
}

template< typename T > VIRTUAL void NetLearner<T>::addPostEpochAction( PostEpochAction *action ) {
    postEpochActions.push_back( action );
}
template< typename T > void NetLearner<T>::learn( float learningRate ) {
    learn( learningRate, 1.0f );
}

template< typename T > void NetLearner<T>::learn( float learningRate, float annealLearningRate ) {
    BatchLearner<T> batchLearner( net );
    Timer timer;
    for( int epoch = startEpoch; epoch <= numEpochs; epoch++ ) {
        float annealedLearningRate = learningRate * pow( annealLearningRate, epoch );
        EpochResult epochResult = batchLearner.runEpochFromLabels( annealedLearningRate, batchSize, Ntrain, trainData, trainLabels );
        if( dumpTimings ) {
            StatefulTimer::dump(true);
        }
//        cout << "-----------------------" << endl;
        cout << endl;
        timer.timeCheck("after epoch " + toString(epoch ) );
        cout << "annealed learning rate: " << annealedLearningRate << " training loss: " << epochResult.loss << endl;
        cout << " train accuracy: " << epochResult.numRight << "/" << Ntrain << " " << (epochResult.numRight * 100.0f/ Ntrain) << "%" << std::endl;
        int testNumRight = batchLearner.test( batchSize, Ntest, testData, testLabels );
        cout << "test accuracy: " << testNumRight << "/" << Ntest << " " << (testNumRight * 100.0f / Ntest ) << "%" << endl;
        timer.timeCheck("after tests");
        for( vector<PostEpochAction *>::iterator it = postEpochActions.begin(); it != postEpochActions.end(); it++ ) {
            (*it)->run( epoch );
        }
    }
}

template class NetLearner<unsigned char>;
template class NetLearner<float>;

