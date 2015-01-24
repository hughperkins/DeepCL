//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>

#include "test/NorbLoader.h"
#include "BoardHelper.h"
#include "Timer.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "stringhelper.h"
#include "FileHelper.h"
#include "StatefulTimer.h"
#include "WeightsPersister.h"
#include "test/NormalizationHelper.h"

using namespace std;

//void getStats( unsigned char *data, int N, int numPlanes, int boardSize, float *p_mean, float *p_stdDev ) {
//    // get mean of the dataset, and stddev
////    float thismax = 0;
//    float sum = 0;
//    int linearLength = N * numPlanes * boardSize * boardSize;
//    for( int i = 0; i < linearLength; i++ ) {
//        int thisValue = (int)data[i];
//        sum += thisValue;
//    }
//    float mean = sum / linearLength;

//    float sumSquaredDiff = 0;
//    for( int i = 0; i < linearLength; i++ ) {
//        int thisValue = (int)data[i];
//        float diffFromMean = thisValue - mean;
//        float diffSquared = diffFromMean * diffFromMean;
//        sumSquaredDiff += diffSquared;
//    }
//    float stdDev = sqrt( sumSquaredDiff / ( linearLength - 1 ) );

//    *p_mean = mean;
//    *p_stdDev = stdDev;
//}

//void normalize( float *data, int N, int numPlanes, int boardSize, double mean, double stdDev ) {
//    int linearLength = N * numPlanes * boardSize * boardSize;
//    for( int i = 0; i < linearLength; i++ ) {
//        data[i] = ( data[i] - mean ) / stdDev;
//    }
//}

class Config {
public:
    string dataDir = "../data/norb";
    string trainSet = "training-shuffled";
    string testSet = "testing-sampled";
    int numTrain = 1000;
    int numTest = 1000;
    int batchSize = 128;
    int numEpochs = 12;
    int numFilters = 16;
    int numLayers = 1;
    int padZeros = 0;
    int filterSize = 5;
    int restartable = 0;
    string restartableFilename = "weights.dat";
    float learningRate = 0.0001f;
    int biased = 1;
    string resultsFilename = "results.txt";
    Config() {
    }
};

float printAccuracy( string name, NeuralNet *net, unsigned char *boards, int *labels, int batchSize, int N, int numPlanes, int boardSize, float mean, float stdDev ) {
    int testNumRight = 0;
    net->setBatchSize( batchSize );
    int numBatches = (N + batchSize - 1 ) / batchSize;
    int inputCubeSize = numPlanes * boardSize * boardSize;
    float *batchData = new float[ batchSize * inputCubeSize ];
    for( int batch = 0; batch < numBatches; batch++ ) {
        int batchStart = batch * batchSize;
        int thisBatchSize = batchSize;
        if( batch == numBatches - 1 ) {
            thisBatchSize = N - batchStart;
            net->setBatchSize( thisBatchSize );
        }
        const int batchInputSize = thisBatchSize * inputCubeSize;
        unsigned char *thisBatchData = boards + batchStart * inputCubeSize;
        for( int i = 0; i < batchInputSize; i++ ) {
            batchData[i] = thisBatchData[i];
        }
        NormalizationHelper::normalize( batchData, batchInputSize, mean, stdDev );
        net->propagate( batchData );
        float const*results = net->getResults();
        int thisnumright = net->calcNumRight( &(labels[batchStart]) );
        testNumRight += thisnumright;
    }
    float accuracy = ( testNumRight * 100.0f / N );
    cout << name << " overall: " << testNumRight << "/" << N << " " << accuracy << "%" << endl;
    delete[] batchData;
    return accuracy;
}

void go(Config config) {
    Timer timer;

    int Ntrain;
    int Ntest;
    int numPlanes;
    int boardSize;

    unsigned char *trainData = NorbLoader::loadImages( config.dataDir + "/" + config.trainSet + "-dat.mat", &Ntrain, &numPlanes, &boardSize, config.numTrain );
    unsigned char *testData = NorbLoader::loadImages( config.dataDir + "/" + config.testSet + "-dat.mat", &Ntest, &numPlanes, &boardSize, config.numTest );
    int *trainLabels = NorbLoader::loadLabels( config.dataDir + "/" + config.trainSet + "-cat.mat", Ntrain );
    int *testLabels = NorbLoader::loadLabels( config.dataDir + "/" + config.testSet + "-cat.mat", Ntest );
    timer.timeCheck("after load images");

    const int inputCubeSize = numPlanes * boardSize * boardSize;
    float mean;
    float stdDev;
    NormalizationHelper::getStats( trainData, Ntrain * inputCubeSize, &mean, &stdDev );
    cout << " board stats mean " << mean << " stdDev " << stdDev << endl;
    timer.timeCheck("after getting stats");

    const int numToTrain = config.numTrain;
    const int batchSize = config.batchSize;
    NeuralNet *net = NeuralNet::maker()->planes(numPlanes)->boardSize(boardSize)->instance();
    net->convolutionalMaker()->numFilters(8)->filterSize(5)->relu()->biased()->insert();
    net->poolingMaker()->poolingSize(4)->insert();
    net->convolutionalMaker()->numFilters(24)->filterSize(6)->relu()->biased()->insert();
    net->poolingMaker()->poolingSize(3)->insert();
    net->fullyConnectedMaker()->numPlanes(5)->boardSize(1)->linear()->biased()->insert();
    net->softMaxLossMaker()->insert();
    net->setBatchSize(config.batchSize);
    net->print();

    if( config.restartable ) {
        WeightsPersister::loadWeights( config.restartableFilename, net );
    }

    timer.timeCheck("before learning start");
    StatefulTimer::timeCheck("START");
    int numBatches = ( config.numTrain + batchSize - 1 ) / batchSize;
    float *batchData = new float[ config.batchSize * inputCubeSize ];
    for( int epoch = 0; epoch < config.numEpochs; epoch++ ) {
        int trainNumRight = 0;
        int thisBatchSize = batchSize;
        net->setBatchSize( thisBatchSize );
        float loss = 0;
        int numRight = 0;
        for( int batch = 0; batch < numBatches; batch++ ) {
            if( batch == numBatches - 1 ) {
                thisBatchSize = config.numTrain - (numBatches - 1) * batchSize;
                net->setBatchSize( thisBatchSize );
            }
            int batchStart = batchSize * batch;
            const int batchInputSize = thisBatchSize * inputCubeSize;
            unsigned char *thisBatchData = trainData + batchStart * inputCubeSize;
            for( int i = 0; i < batchInputSize; i++ ) {
                batchData[i] = thisBatchData[i];
            }
            NormalizationHelper::normalize( batchData, batchInputSize, mean, stdDev );
            net->learnBatchFromLabels( config.learningRate, batchData, &(trainLabels[batchStart]) );
            loss += net->calcLossFromLabels( &(trainLabels[batchStart]) );
            numRight += net->calcNumRight( &(trainLabels[batchStart]) );
        }
        StatefulTimer::dump(true);
        cout << "       loss L: " << loss << endl;
        timer.timeCheck("after epoch " + toString(epoch) );
//        net->print();
        std::cout << "train accuracy: " << numRight << "/" << numToTrain << " " << (numRight * 100.0f/ numToTrain) << "%" << std::endl;
        printAccuracy( "test", net, testData, testLabels, batchSize, config.numTest, numPlanes, boardSize, mean, stdDev );
        timer.timeCheck("after tests");
        if( config.restartable ) {
            WeightsPersister::persistWeights( config.restartableFilename, net );
        }
    }

    delete net;

    delete[] trainData;
    delete[] testData;
    delete[] testLabels;
    delete[] trainLabels;

}

int main( int argc, char *argv[] ) {
    Config config;
    if( argc == 2 && ( string(argv[1]) == "--help" || string(argv[1]) == "--?" || string(argv[1]) == "-?" || string(argv[1]) == "-h" ) ) {
        cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
        cout << "Possible key=value pairs:" << endl;
        cout << "    datadir=[data directory] (" << config.dataDir << ")" << endl;
        cout << "    trainset=[training-shuffled|testing-sampled|other set name] (" << config.trainSet << ")" << endl;
        cout << "    testset=[training-shuffled|testing-sampled|other set name] (" << config.testSet << ")" << endl;
        cout << "    numtrain=[num training examples] (" << config.numTrain << ")" << endl;
        cout << "    numtest=[num test examples] (" << config.numTest << ")" << endl;
        cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
        cout << "    numepochs=[number epochs] (" << config.numEpochs << ")" << endl;
        cout << "    numlayers=[number convolutional layers] (" << config.numLayers << ")" << endl;
        cout << "    numfilters=[number filters] (" << config.numFilters << ")" << endl;
        cout << "    filtersize=[filter size] (" << config.filterSize << ")" << endl;
        cout << "    biased=[0|1] (" << config.biased << ")" << endl;
        cout << "    padzeros=[0|1] (" << config.padZeros << ")" << endl;
        cout << "    learningrate=[learning rate, a float value] (" << config.learningRate << ")" << endl;
        cout << "    restartable=[weights are persistent?] (" << config.restartable << ")" << endl;
        cout << "    restartablefilename=[filename to store weights] (" << config.restartableFilename << ")" << endl;
        cout << "    resultsfilename=[filename to store results] (" << config.resultsFilename << ")" << endl;
    } 
    for( int i = 1; i < argc; i++ ) {
       vector<string> splitkeyval = split( argv[i], "=" );
       if( splitkeyval.size() != 2 ) {
          cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
          exit(1);
       } else {
           string key = splitkeyval[0];
           string value = splitkeyval[1];
           if( key == "datadir" ) config.dataDir = value;
           if( key == "trainset" ) config.trainSet = value;
           if( key == "testset" ) config.testSet = value;
           if( key == "numtrain" ) config.numTrain = atoi(value);
           if( key == "numtest" ) config.numTest = atoi(value);
           if( key == "batchsize" ) config.batchSize = atoi(value);
           if( key == "numepochs" ) config.numEpochs = atoi(value);
           if( key == "biased" ) config.biased = atoi(value);
           if( key == "numfilters" ) config.numFilters = atoi(value);
           if( key == "numlayers" ) config.numLayers = atoi(value);
           if( key == "padzeros" ) config.padZeros = atoi(value);
           if( key == "filtersize" ) config.filterSize = atoi(value);
           if( key == "learningrate" ) config.learningRate = atof(value);
           if( key == "restartable" ) config.restartable = atoi(value);
           if( key == "restartablefilename" ) config.restartableFilename = value;
           if( key == "resultsfilename" ) config.resultsFilename = value;
       }
    }
    go( config );
}


