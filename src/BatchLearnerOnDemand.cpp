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

#include "BatchLearnerOnDemand.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

//template< typename T>
//void NetLearnLabeledBatch<T>::run( Trainable *net, T *batchData, int const*batchLabels ) {
//    net->learnBatchFromLabels( learningRate, batchData, batchLabels );
//}

//template< typename T>
//void NetPropagateBatch<T>::run( Trainable *net, T *batchData, int const*batchLabels ) {
//    net->propagate( batchData );
//}

template< typename T > BatchLearnerOnDemand<T>::BatchLearnerOnDemand( Trainable *net ) :
    net( net ) {
}

template< typename T > EpochResult BatchLearnerOnDemand<T>::batchedNetAction( std::string filepath, int batchSize, int N, NetAction<T> *netAction ) {
    int numRight = 0;
    float loss = 0;
    net->setBatchSize( batchSize );
    int numBatches = (N + batchSize - 1 ) / batchSize;
    int inputCubeSize = net->getInputCubeSize();
    T *dataBuffer = new T[ batchSize * inputCubeSize ];
    int *labelsBuffer = new int[ batchSize * inputCubeSize ];
    int thisBatchSize = batchSize;
    for( int batch = 0; batch < numBatches; batch++ ) {
        int batchStart = batch * batchSize;
        if( batch == numBatches - 1 ) {
            thisBatchSize = N - batchStart;
            net->setBatchSize( thisBatchSize );
        }
//        cout << "batch " << batch << " batchSize " << batchSize << " N " << N << " thisbatchsize " << thisBatchSize << endl;
        GenericLoader::load( filepath, dataBuffer, labelsBuffer, batchStart, thisBatchSize );
//        cout << "loaded from file " << endl;
        netAction->run( net, dataBuffer, labelsBuffer );
//        cout << "trained" << endl;
        loss += net->calcLossFromLabels( labelsBuffer );
        numRight += net->calcNumRight( labelsBuffer );
    }
    EpochResult epochResult( loss, numRight );
    delete[] dataBuffer;
    delete[] labelsBuffer;
    return epochResult;
}

template< typename T > int BatchLearnerOnDemand<T>::test( std::string filepath, int batchSize, int Ntest ) {
    net->setTraining( false );
    NetAction<T> *action = new NetPropagateBatch<T>();
    int numRight = batchedNetAction( filepath, batchSize, Ntest, action ).numRight;
    delete action;
    return numRight;
}

template< typename T > EpochResult BatchLearnerOnDemand<T>::runEpochFromLabels( float learningRate, std::string filepath, int batchSize, int Ntrain ) {
    net->setTraining( true );
    NetAction<T> *action = new NetLearnLabeledBatch<T>( learningRate );
    EpochResult epochResult = batchedNetAction( filepath, batchSize, Ntrain, action );
    delete action;
    return epochResult;
}

//template< typename T > float BatchLearnerOnDemand<T>::runEpochFromExpected( float learningRate, std::string filepath, int batchSize, int N, T *data, float *expectedResults ) {
//    net->setTraining( true );
//    float loss = 0;
//    net->setBatchSize( batchSize );
//    const int numBatches = (N + batchSize - 1 ) / batchSize;
//    const int inputCubeSize = net->getInputCubeSize();
//    const int outputCubeSize = net->getOutputCubeSize();
//    for( int batch = 0; batch < numBatches; batch++ ) {
//        int batchStart = batch * batchSize;
//        if( batch == numBatches - 1 ) {
//            net->setBatchSize( N - batchStart );
//        }
//        net->learnBatch( learningRate, &(data[ batchStart * inputCubeSize ]), &(expectedResults[batchStart * outputCubeSize]) );
//        loss += net->calcLoss( &( expectedResults[batchStart * outputCubeSize]) );
//    }
//    return loss;
//}

template class BatchLearnerOnDemand<unsigned char>;
//template class BatchLearnerOnDemand<float>;

