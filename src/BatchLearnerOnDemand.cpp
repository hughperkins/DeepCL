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

BatchLearnerOnDemand::BatchLearnerOnDemand( Trainable *net ) :
    net( net ) {
}

class OnDemandBatcher {
public:
    Trainable *net;
    BatchLearner batchLearner;
    NetAction *netAction; // NOt owned by us, dont delete
    std::string filepath;
    int N;
    int fileReadBatches;
    int batchSize;
    int fileBatchSize;
    int inputCubeSize;
    int numFileBatches;

    float *dataBuffer;
    int *labelsBuffer;

    bool learningDone;
    int numRight;
    float loss;
    int nextFileBatch;

    OnDemandBatcher( Trainable *net, NetAction *netAction, 
                std::string filepath, int N, int fileReadBatches, int batchSize ) :
                net( net ),
                batchLearner( net ),
                netAction( netAction ),
                filepath( filepath ),
                N( N ),
                fileReadBatches( fileReadBatches ),
                batchSize( batchSize ),
                fileBatchSize( batchSize * fileReadBatches ),
                inputCubeSize( net->getInputCubeSize() )
            {
        fileBatchSize = fileBatchSize > N ? N : fileBatchSize;
        numFileBatches = ( N + fileBatchSize - 1 ) / fileBatchSize;
        dataBuffer = new float[ fileBatchSize * inputCubeSize ];
        labelsBuffer = new int[ fileBatchSize * inputCubeSize ];
        reset();
    }
    virtual ~OnDemandBatcher() {
        delete[] dataBuffer;
        delete[] labelsBuffer;
    }
    void reset() {
        numRight = 0;
        loss = 0;
        nextFileBatch = 0;
        learningDone = false;
    }
    bool tick() {
        int fileBatch = nextFileBatch;
        int thisFileBatchSize = fileBatchSize;
        int fileBatchStart = fileBatch * fileBatchSize;
        if( fileBatch == numFileBatches - 1 ) {
            thisFileBatchSize = N - fileBatchStart;
        }
//        cout << "batchlearnerondemand, read data... filebatchstart=" << fileBatchStart << endl;
        GenericLoader::load( filepath, dataBuffer, labelsBuffer, fileBatchStart, thisFileBatchSize );
        EpochResult epochResult = batchLearner.batchedNetAction( batchSize, thisFileBatchSize, dataBuffer, labelsBuffer, netAction );
        loss += epochResult.loss;
        numRight += epochResult.numRight;

        nextFileBatch++;
        if( nextFileBatch == numFileBatches ) {
            learningDone = true;
        }
        return !learningDone;
    }
    EpochResult run() {
        if( learningDone ) {
            reset();
        }
        while( !learningDone ) {
            tick();
        }
        EpochResult epochResult( loss, numRight );
        return epochResult;
    }
};

EpochResult BatchLearnerOnDemand::runBatchedNetAction( std::string filepath, int fileReadBatches, int batchSize, int N, NetAction *netAction ) {
    OnDemandBatcher onDemandBatcher(net, netAction, filepath, N, fileReadBatches, batchSize );
    return onDemandBatcher.run();
}

int BatchLearnerOnDemand::test( std::string filepath, int fileReadBatches, int batchSize, int Ntest ) {
    net->setTraining( false );
    NetAction *action = new NetPropagateBatch();
    int numRight = runBatchedNetAction( filepath, fileReadBatches, batchSize, Ntest, action ).numRight;
    delete action;
    return numRight;
}

EpochResult BatchLearnerOnDemand::runEpochFromLabels( float learningRate, std::string filepath, int fileReadBatches, int batchSize, int Ntrain ) {
    net->setTraining( true );
    NetAction *action = new NetLearnLabeledBatch( learningRate );
    EpochResult epochResult = runBatchedNetAction( filepath, fileReadBatches, batchSize, Ntrain, action );
    delete action;
    return epochResult;
}


