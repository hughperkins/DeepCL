// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "StatefulTimer.h"
#include "Timer.h"
#include "BatchLearner.h"
#include "NeuralNet.h"

#include "NetLearner.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

template< typename T > NetLearner<T>::NetLearner( NeuralNet *net ) :
        net( net ) {
    batchSize = 128;
    annealLearningRate = 1.0f;
    translate = 0;
    scale = 1.0f;
    numEpochs = 12;
    startEpoch = 0;
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
    setSchedule( numEpochs, 0 );
}

template< typename T > void NetLearner<T>::setSchedule( int numEpochs, int startEpoch ) {
    this->numEpochs = numEpochs;
    this->startEpoch = startEpoch;
}

template< typename T > void NetLearner<T>::setNormalize( float translate, float scale ) {
    this->translate = translate;
    this->scale = scale;
}

template< typename T > void NetLearner<T>::setBatchSize( int batchSize ) {
    this->batchSize = batchSize;
}

template< typename T > VIRTUAL NetLearner<T>::~NetLearner() {
    for( vector<PostEpochAction *>::iterator it = postEpochActions.begin(); it != postEpochActions.end(); it++ ) {
        delete (*it);
    }
}

template< typename T > VIRTUAL void NetLearner<T>::addPostEpochAction( PostEpochAction *action ) {
    postEpochActions.push_back( action );
}
template< typename T > void NetLearner<T>::learn( float learningRate ) {
    learn( learningRate, 1.0f );
}

template< typename T > void NetLearner<T>::learn( float learningRate, float annealLearningRate ) {
    BatchLearner<T> batchLearner( net, translate, scale );
    Timer timer;
    for( int epoch = startEpoch; epoch < numEpochs; epoch++ ) {
        float annealedLearningRate = learningRate * pow( annealLearningRate, epoch );
        cout << "Annealed learning rate: " << annealedLearningRate << endl;
        EpochResult epochResult = batchLearner.runEpochFromLabels( annealedLearningRate, batchSize, Ntrain, trainData, trainLabels );
        StatefulTimer::dump(true);
        cout << "       loss L: " << epochResult.loss << endl;
        timer.timeCheck("after epoch " + toString(epoch) );
        std::cout << "train accuracy: " << epochResult.numRight << "/" << Ntrain << " " << (epochResult.numRight * 100.0f/ Ntrain) << "%" << std::endl;
        int testNumRight = batchLearner.test( batchSize, Ntest, testData, testLabels );
        cout << "test accuracy: " << testNumRight << "/" << Ntest << " " << (testNumRight * 100.0f / Ntest ) << "%" << endl;
        timer.timeCheck("after tests");
        for( vector<PostEpochAction *>::iterator it = postEpochActions.begin(); it != postEpochActions.end(); it++ ) {
            (*it)->run( epoch );
        }
//        if( config.restartable ) {
//            WeightsPersister::persistWeights( config.restartableFilename, config.getTrainingString(), net, epoch + 1, 0, annealedLearningRate, 0, 0 );
//        }
    }
}

template class NetLearner<unsigned char>;
template class NetLearner<float>;

