// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>

#include "NorbLoader.h"
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

/* [[[cog
    # cog.outl('// generated using cog:')
    strings = { 
        'dataDir': '../data/norb', 
        'trainSet': 'training-shuffled',
        'testSet': 'testing-sampled',
        'netDef': '8C5-MP4-24C6-MP3-80C6-10N',
        'restartableFilename': 'weights.dat'
    }
    ints = {
        'numTrain': 0,
        'numTest': 0,
        'batchSize': 128,
        'numEpochs': 20,
        'restartable': 1
    }
    floats = {
        'learningRate': 0.0001,
        'annealLearningRate': 0.95
    }
    descriptions = {
        'datadir': 'data directory',
        'trainset': '[training-shuffled|testing-sampled|other set name]',
        'testset': '[training-shuffled|testing-sampled|other set name]',
        'numtrain': 'num training examples',
        'numtest': 'num test examples]',
        'batchsize': 'batch size',
        'numepochs': 'number epochs',
        'netdef': 'network definition',
        'learningrate': 'learning rate, a float value',
        'anneallearningrate': 'multiply learning rate by this, each epoch',
        'restartable': 'weights are persistent?',
        'restartablefilename': 'filename to store weights'
    }
*///]]]
// [[[end]]]

class Config {
public:
    // [[[cog
    // cog.outl('// generated using cog:')
    // for astring in strings.keys():
    //    cog.outl( 'string ' + astring + ' = "";')
    // for anint in ints.keys():
    //    cog.outl( 'int ' + anint + ' = 0;')
    // for name in floats.keys():
    //    cog.outl( 'float ' + name + ' = 0;')
    // ]]]
    // generated using cog:
    string netDef = "";
    string dataDir = "";
    string testSet = "";
    string restartableFilename = "";
    string trainSet = "";
    int batchSize = 0;
    int numTest = 0;
    int restartable = 0;
    int numTrain = 0;
    int numEpochs = 0;
    float learningRate = 0;
    float annealLearningRate = 0;
    // [[[end]]]

    Config() {
        netDef = "8C5-MP4-24C6-MP3-80C6-10N";
        dataDir = "../data/norb";
        testSet = "testing-sampled";
        restartableFilename = "weights.dat";
        trainSet = "training-shuffled";
        batchSize = 128;
        numTest = 0;
        restartable = 0;
        numTrain = 0;
        numEpochs = 20;
        learningRate = 0.0001f;
        annealLearningRate = 0.95f;
    }
    string getTrainingString() {
        string configString = "";
        configString += "netDef=" + netDef + " dataDir=" + dataDir + " trainSet=" + trainSet;
        return configString;
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
    NormalizationHelper::getMeanAndStdDev( trainData, Ntrain * inputCubeSize, &mean, &stdDev );
    cout << " board stats mean " << mean << " stdDev " << stdDev << endl;
    timer.timeCheck("after getting stats");

    const int numToTrain = Ntrain;
    const int batchSize = config.batchSize;
    NeuralNet *net = NeuralNet::maker()->planes(numPlanes)->boardSize(boardSize)->instance();

    string netDefLower = toLower( config.netDef );
    vector<string> splitNetDef = split( netDefLower, "-" );
    for( int i = 0; i < splitNetDef.size(); i++ ) {
        string thisLayerDef = splitNetDef[i];
        if( thisLayerDef.find("c") != string::npos ) {
            vector<string> splitConvDef = split( thisLayerDef, "c" );
            int numFilters = atoi( splitConvDef[0] );
            int filterSize = atoi( splitConvDef[1] );
            net->convolutionalMaker()->numFilters(numFilters)->filterSize(filterSize)->relu()->biased()->insert();
        } else if( thisLayerDef.find("mp") != string::npos ) {
            vector<string> splitPoolDef = split( thisLayerDef, "mp" );
            int poolingSize = atoi( splitPoolDef[1] );
            net->poolingMaker()->poolingSize(poolingSize)->insert();
        } else if( thisLayerDef.find("n") != string::npos ) {
            vector<string> fullDef = split( thisLayerDef, "n" );
            int numPlanes = atoi( fullDef[0] );
            if( i == splitNetDef.size() - 1 ) {
                net->fullyConnectedMaker()->numPlanes(numPlanes)->boardSize(1)->linear()->biased()->insert();
            } else {
                net->fullyConnectedMaker()->numPlanes(numPlanes)->boardSize(1)->tanh()->biased()->insert();
            }
        } else {
            throw runtime_error("network definition " + thisLayerDef + " not recognised" );
        }
    }
    net->softMaxLossMaker()->insert();
    net->setBatchSize(config.batchSize);
    net->print();

    bool afterRestart = false;
    int restartEpoch = 0;
    int restartBatch = 0;
    float restartAnnealedLearningRate = 0;
    int restartNumRight = 0;
    float restartLoss = 0;
    if( config.restartable ) {
        afterRestart = WeightsPersister::loadWeights( config.restartableFilename, config.getTrainingString(), net, &restartEpoch, &restartBatch, &restartAnnealedLearningRate, &restartNumRight, &restartLoss );
        if( !afterRestart && FileHelper::exists( config.restartableFilename ) ) {
            cout << "Weights file " << config.restartableFilename << " exists, but doesnt match training options provided => aborting" << endl;
            cout << "Please either check the training options, or choose a weights file that doesnt exist yet" << endl;
            return;
        }
    }

    timer.timeCheck("before learning start");
    StatefulTimer::timeCheck("START");
    int numBatches = ( Ntrain + batchSize - 1 ) / batchSize;
    float *batchData = new float[ config.batchSize * inputCubeSize ];
    float annealedLearningRate = afterRestart ? restartAnnealedLearningRate : config.learningRate;
    for( int epoch = afterRestart ? restartEpoch : 0; epoch < config.numEpochs; epoch++ ) {
        cout << "Annealed learning rate: " << annealedLearningRate << endl;
//        int trainNumRight = 0;
        int thisBatchSize = batchSize;
        net->setBatchSize( thisBatchSize );
        float loss = afterRestart ? restartLoss : 0;
        int numRight = afterRestart ? restartNumRight : 0;
        for( int batch = afterRestart ? restartBatch : 0; batch < numBatches; batch++ ) {
            afterRestart = false;
            if( batch == numBatches - 1 ) {
                thisBatchSize = Ntrain - (numBatches - 1) * batchSize;
                net->setBatchSize( thisBatchSize );
            }
            int batchStart = batchSize * batch;
            const int batchInputSize = thisBatchSize * inputCubeSize;
            unsigned char *thisBatchData = trainData + batchStart * inputCubeSize;
            for( int i = 0; i < batchInputSize; i++ ) {
                batchData[i] = thisBatchData[i];
            }
            NormalizationHelper::normalize( batchData, batchInputSize, mean, stdDev );
            net->learnBatchFromLabels( annealedLearningRate, batchData, &(trainLabels[batchStart]) );
            loss += net->calcLossFromLabels( &(trainLabels[batchStart]) );
            numRight += net->calcNumRight( &(trainLabels[batchStart]) );
        }
        StatefulTimer::dump(true);
        cout << "       loss L: " << loss << endl;
        timer.timeCheck("after epoch " + toString(epoch) );
//        net->print();
        std::cout << "train accuracy: " << numRight << "/" << numToTrain << " " << (numRight * 100.0f/ Ntrain) << "%" << std::endl;
        printAccuracy( "test", net, testData, testLabels, batchSize, Ntest, numPlanes, boardSize, mean, stdDev );
        timer.timeCheck("after tests");
        annealedLearningRate *= config.annealLearningRate;
        if( config.restartable ) {
            WeightsPersister::persistWeights( config.restartableFilename, config.getTrainingString(), net, epoch + 1, 0, annealedLearningRate, 0, 0 );
        }
    }

    delete net;

    delete[] trainData;
    delete[] testData;
    delete[] testLabels;
    delete[] trainLabels;

}

void printUsage( char *argv[], Config config ) {
    cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
    cout << endl;
    cout << "Possible key=value pairs:" << endl;
    // [[[cog
    // cog.outl('// generated using cog:')
    // for name in strings.keys():
    //    cog.outl( 'cout << "    ' + name.lower() + '=[' + descriptions[name.lower()] + '] (" << config.' + name + ' << ")" << endl;')
    // for name in ints.keys():
    //    cog.outl( 'cout << "    ' + name.lower() + '=[' + descriptions[name.lower()] + '] (" << config.' + name + ' << ")" << endl;')
    // for name in floats.keys():
    //    cog.outl( 'cout << "    ' + name.lower() + '=[' + descriptions[name.lower()] + '] (" << config.' + name + ' << ")" << endl;')
    // ]]]
    // generated using cog:
    cout << "    netdef=[network definition] (" << config.netDef << ")" << endl;
    cout << "    datadir=[data directory] (" << config.dataDir << ")" << endl;
    cout << "    testset=[[training-shuffled|testing-sampled|other set name]] (" << config.testSet << ")" << endl;
    cout << "    restartablefilename=[filename to store weights] (" << config.restartableFilename << ")" << endl;
    cout << "    trainset=[[training-shuffled|testing-sampled|other set name]] (" << config.trainSet << ")" << endl;
    cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
    cout << "    numtest=[num test examples]] (" << config.numTest << ")" << endl;
    cout << "    restartable=[weights are persistent?] (" << config.restartable << ")" << endl;
    cout << "    numtrain=[num training examples] (" << config.numTrain << ")" << endl;
    cout << "    numepochs=[number epochs] (" << config.numEpochs << ")" << endl;
    cout << "    learningrate=[learning rate, a float value] (" << config.learningRate << ")" << endl;
    cout << "    anneallearningrate=[multiply learning rate by this, each epoch] (" << config.annealLearningRate << ")" << endl;
    // [[[end]]]
}

int main( int argc, char *argv[] ) {
    Config config;
    if( argc == 2 && ( string(argv[1]) == "--help" || string(argv[1]) == "--?" || string(argv[1]) == "-?" || string(argv[1]) == "-h" ) ) {
        printUsage( argv, config );
    } 
    for( int i = 1; i < argc; i++ ) {
        vector<string> splitkeyval = split( argv[i], "=" );
        if( splitkeyval.size() != 2 ) {
          cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
          exit(1);
        } else {
            string key = splitkeyval[0];
            string value = splitkeyval[1];
            // [[[cog
            // cog.outl('// generated using cog:')
            // cog.outl('if( false ) {')
            // for name in strings.keys():
            //    cog.outl( '} else if( key == "' + name.lower() + '" ) {')
            //    cog.outl( '    config.' + name + ' = value;')
            // for name in ints.keys():
            //    cog.outl( '} else if( key == "' + name.lower() + '" ) {')
            //    cog.outl( '    config.' + name + ' = atoi( value );')
            // for name in floats.keys():
            //    cog.outl( '} else if( key == "' + name.lower() + '" ) {')
            //    cog.outl( '    config.' + name + ' = atof( value );')
            // ]]]
            // generated using cog:
            if( false ) {
            } else if( key == "netdef" ) {
                config.netDef = value;
            } else if( key == "datadir" ) {
                config.dataDir = value;
            } else if( key == "testset" ) {
                config.testSet = value;
            } else if( key == "restartablefilename" ) {
                config.restartableFilename = value;
            } else if( key == "trainset" ) {
                config.trainSet = value;
            } else if( key == "batchsize" ) {
                config.batchSize = atoi( value );
            } else if( key == "numtest" ) {
                config.numTest = atoi( value );
            } else if( key == "restartable" ) {
                config.restartable = atoi( value );
            } else if( key == "numtrain" ) {
                config.numTrain = atoi( value );
            } else if( key == "numepochs" ) {
                config.numEpochs = atoi( value );
            } else if( key == "learningrate" ) {
                config.learningRate = atof( value );
            } else if( key == "anneallearningrate" ) {
                config.annealLearningRate = atof( value );
            // [[[end]]]
            } else {
                cout << endl;
                cout << "Error: key '" << key << "' not recognised" << endl;
                cout << endl;
                printUsage( argv, config );
                cout << endl;
                return -1;
            }
        }
    }
    go( config );
}


