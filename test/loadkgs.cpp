// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

//#include "FileHelper.h"
#include "Timer.h"
#include "stringhelper.h"
#include "NeuralNet.h"
//#include "AccuracyHelper.h"
//#include "BoardHelper.h"
#include "InputLayerMaker.h"
#include "BatchLearner.h"
#include "KgsLoader.h"

using namespace std;

int main(int argc, char *argv[] ) {
    const float learningRate = 0.0001f;
    const int batchSize = 128;
//    const int labelGrouping = 1;

//    const int numLabelRows = ((19+labelGrouping-1)/labelGrouping);
//    const int numLabels = numLabelRows * numLabelRows;
//    cout << " labelgrouping " << labelGrouping << " numLabelRows " << numLabelRows << " numLabels " << numLabels << endl;
//return -1;
    Timer timer;
    std::string trainFilepath = "../data/kgsgo/kgsgo-train10k.dat";
    std::string testFilepath = "../data/kgsgo/kgsgo-test.dat";
//    std::string dataFilePath = "/home/user/git/kgsgo-dataset-preprocessor/data/kgsgo.dat";
//    long trainFilesize = FileHelper::getFilesize( trainFilepath );

    int numPlanes = 8;
    int boardSize = 19;

    int Ntrain = KgsLoader::getNumRecords( trainFilepath );
    int Ntest = KgsLoader::getNumRecords( testFilepath );
    int testInputSize = Ntest * numPlanes * boardSize * boardSize;
    cout << "testINputSize " << testInputSize << endl;
    unsigned char *testData = new unsigned char[ testInputSize ];
    int *testLabels = new int[Ntest];
    KgsLoader::loadKgs( testFilepath, &numPlanes, &boardSize, testData, testLabels, 0, Ntest );

//    timer.timeCheck("read file, size " + toString(trainFilesize/1024/1024) + "MB");
//    int i = 0;
//    long pos = 0;
//    const int boardSize = 19;
    const int boardSizeSquared = boardSize * boardSize;
//    const int recordSize = 2 + 2 + boardSizeSquared;
//    cout << "recordsize: " << recordSize << endl;
//    int inputPlanes = 2;
    unsigned char *trainData = new unsigned char[batchSize * numPlanes * boardSizeSquared ];
    int *trainLabels = new int[batchSize ];
//    int N = (int)(fileSize / recordSize);
//    cout << "num records: " << N << endl;
//    int numBatches = min( 8, N / batchSize );

    NeuralNet *net = NeuralNet::maker()->instance();
//    net->inputMaker<unsigned char>()->numPlanes(numPlanes)->boardSize(boardSize)->insert();
    net->addLayer( InputLayerMaker<unsigned char>::instance()->numPlanes(numPlanes)->boardSize(boardSize) );
    net->addLayer( NormalizationLayerMaker::instance()->translate(-0.3f)->scale(1.0f) );
    for( int i = 0; i < 2; i++ ) {
        net->addLayer( ConvolutionalMaker::instance()->numFilters(16)->filterSize(5)->relu()->biased()->padZeros() );
//        net->poolingMaker()->poolingSize(2)->insert();
    }
    net->addLayer( FullyConnectedMaker::instance()->numPlanes(boardSizeSquared)->boardSize(1)->linear()->biased() );
    net->softMaxLossMaker()->insert();
    net->print();

    int numBatches = ( Ntrain + batchSize - 1 ) / batchSize;
    numBatches = 100;
    Ntrain = numBatches * batchSize;
//    int *labels = new int[ batchSize ];
    for( int epoch = 0; epoch < 20000; epoch++ ) {
        net->setBatchSize( batchSize );
        int thisBatchSize = batchSize;
        float epochLoss = 0;
        int epochTrainRight = 0;
        for( int batch = 0; batch < numBatches; batch++ ) {
            const int batchStart = batch * batchSize;
            if( batch == numBatches - 1 ) {
                thisBatchSize = Ntrain - batchStart;
                net->setBatchSize( thisBatchSize );
            }
            KgsLoader::loadKgs( trainFilepath, &numPlanes, &boardSize, trainData, trainLabels, batchStart, thisBatchSize );
//            net->propagate( images );
//            int numRight = net->calcNumRight( trainLabels );
            net->learnBatchFromLabels( learningRate, trainData, trainLabels );
            float loss = net->calcLossFromLabels( trainLabels );
            int numRight = net->calcNumRight( trainLabels );
            epochLoss += loss;
            epochTrainRight += numRight;
//            cout << "batch " << batch << " train loss: " << loss << " train accuracy " << numRight << "/" << batchSize << " " << ( numRight * 100.0f / batchSize ) << endl;
//            net->propagate( testData );
//            int testRight = net->calcNumRight( testLabels );
        }
        timer.timeCheck("epoch " + toString( epoch ) );
        StatefulTimer::dump(true);
        cout << "epoch train totals: " << epochTrainRight << "/" << Ntrain << " " << ( epochTrainRight * 100.0f / Ntrain ) << endl;
        BatchLearner<unsigned char>batchLearner( net );
        int testRight = batchLearner.test( batchSize, Ntest, testData, testLabels );
        timer.timeCheck("testing");
        cout << "test accuracy " << testRight << "/" << Ntest << " " << ( testRight * 100.0f / Ntest ) << endl;
    }
    return 0;
}

