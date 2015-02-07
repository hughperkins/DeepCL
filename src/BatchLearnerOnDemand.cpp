// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NormalizationHelper.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "Trainable.h"
#include "GenericLoader.h"
#include "BatchLearner.h"

#include "BatchLearnerOnDemand.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

template< typename T > BatchLearnerOnDemand<T>::BatchLearnerOnDemand( Trainable *net ) :
    net( net ) {
}

template< typename T > EpochResult BatchLearnerOnDemand<T>::batchedNetAction( std::string filepath, int fileReadBatches, int batchSize, int N, NetAction<T> *netAction ) {
    int numRight = 0;
    float loss = 0;
    int fileBatchSize = batchSize * fileReadBatches;
    int inputCubeSize = net->getInputCubeSize();
    T *dataBuffer = new T[ fileReadBatches *  batchSize * inputCubeSize ];
    int *labelsBuffer = new int[ fileReadBatches * batchSize * inputCubeSize ];
    int numFileBatches = ( N + fileBatchSize - 1 ) / fileBatchSize;
    int thisFileBatchSize = fileBatchSize;
    BatchLearner<unsigned char> batchLearner( net );
    for( int fileBatch = 0; fileBatch < numFileBatches; fileBatch++ ) {
        int fileBatchStart = fileBatch * fileBatchSize;
        if( fileBatch == numFileBatches - 1 ) {
            thisFileBatchSize = N - fileBatchStart;
        }
        GenericLoader::load( filepath, dataBuffer, labelsBuffer, fileBatchStart, thisFileBatchSize );
        EpochResult epochResult = batchLearner.batchedNetAction( batchSize, fileBatchSize, dataBuffer, labelsBuffer, netAction );
        loss += epochResult.loss;
        numRight += epochResult.numRight;
    }
    EpochResult epochResult( loss, numRight );
    delete[] dataBuffer;
    delete[] labelsBuffer;
    return epochResult;
}

template< typename T > int BatchLearnerOnDemand<T>::test( std::string filepath, int fileReadBatches, int batchSize, int Ntest ) {
    net->setTraining( false );
    NetAction<T> *action = new NetPropagateBatch<T>();
    int numRight = batchedNetAction( filepath, fileReadBatches, batchSize, Ntest, action ).numRight;
    delete action;
    return numRight;
}

template< typename T > EpochResult BatchLearnerOnDemand<T>::runEpochFromLabels( float learningRate, std::string filepath, int fileReadBatches, int batchSize, int Ntrain ) {
    net->setTraining( true );
    NetAction<T> *action = new NetLearnLabeledBatch<T>( learningRate );
    EpochResult epochResult = batchedNetAction( filepath, fileReadBatches, batchSize, Ntrain, action );
    delete action;
    return epochResult;
}

template class BatchLearnerOnDemand<unsigned char>;

