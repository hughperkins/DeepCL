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
        net( net ), 
        learnAction(0),
        testAction(),
        trainBatcher( net, 0, 0, 0, 0, &learnAction ),
        testBatcher( net, 0, 0, 0, 0, &testAction ) {
//    batchSize = 128;
    annealLearningRate = 1.0f;
    numEpochs = 12;
    nextEpoch = 0;
    dumpTimings = false;
//    batcher = 0;
//    learnAction = 0;
    learningDone = false;
    
//    reset();
}

template< typename T > void NetLearner<T>::setTrainingData( int Ntrain, T *trainData, int *trainLabels ) {
    this->trainBatcher.N = Ntrain;
    this->trainBatcher.data = trainData;
    this->trainBatcher.labels = trainLabels;
//    cout << "NetLearner.settrainingdata data=" << (void *)trainData << endl;
}

template< typename T > void NetLearner<T>::setTestingData( int Ntest, T *testData, int *testLabels ) {
    this->testBatcher.N = Ntest;
    this->testBatcher.data = testData;
    this->testBatcher.labels = testLabels;
}

template< typename T > void NetLearner<T>::setSchedule( int numEpochs ) {
    setSchedule( numEpochs, 0 );
}

template< typename T > void NetLearner<T>::setDumpTimings( bool dumpTimings ) {
    this->dumpTimings = dumpTimings;
}

template< typename T > void NetLearner<T>::setSchedule( int numEpochs, int startEpoch ) {
    this->numEpochs = numEpochs;
    this->nextEpoch = startEpoch;
}

template< typename T > void NetLearner<T>::setBatchSize( int batchSize ) {
    this->trainBatcher.batchSize = batchSize;
    this->testBatcher.batchSize = batchSize;
}

template< typename T > VIRTUAL NetLearner<T>::~NetLearner() {
}

template< typename T > VIRTUAL void NetLearner<T>::addPostEpochAction( PostEpochAction *action ) {
    postEpochActions.push_back( action );
}

template< typename T > void NetLearner<T>::reset() {
    learningDone = false;
    nextEpoch = 0;
    net->setTraining( true );
    trainBatcher.reset();
    testBatcher.reset();
    timer.lap();
}

template< typename T > bool NetLearner<T>::tickEpoch() {
    int epoch = nextEpoch;
    cout << "NetLearner.tickEpoch epoch=" << epoch << endl;
    learnAction.learningRate = learningRate * pow( annealLearningRate, epoch );
    trainBatcher.run();
    if( dumpTimings ) {
        StatefulTimer::dump(true);
    }
//        cout << "-----------------------" << endl;
    cout << endl;
    timer.timeCheck("after epoch " + toString(epoch+1) );
    cout << "annealed learning rate: " << learnAction.learningRate << " training loss: " << trainBatcher.loss << endl;
    cout << " train accuracy: " << trainBatcher.numRight << "/" << trainBatcher.N << " " << (trainBatcher.numRight * 100.0f/ trainBatcher.N) << "%" << std::endl;
    testBatcher.run();
    cout << "test accuracy: " << testBatcher.numRight << "/" << testBatcher.N << " " << 
        (testBatcher.numRight * 100.0f / testBatcher.N ) << "%" << endl;
    timer.timeCheck("after tests");

    for( vector<PostEpochAction *>::iterator it = postEpochActions.begin(); it != postEpochActions.end(); it++ ) {
        (*it)->run( epoch );
    }

    nextEpoch++;
    if( nextEpoch == numEpochs ) {
        learningDone = true;
    }
    return !learningDone;
}

template< typename T > void NetLearner<T>::learn( float learningRate ) {
    learn( learningRate, 1.0f );
}
template< typename T > void NetLearner<T>::learn( float learningRate, float annealLearningRate ) {
    this->learningRate = learningRate;
    this->annealLearningRate = annealLearningRate;
    if( learningDone ) {
        reset();
    }
    while( !learningDone ) {
        tickEpoch();
    }
}

template class NetLearner<unsigned char>;
template class NetLearner<float>;

