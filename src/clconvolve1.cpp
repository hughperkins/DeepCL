// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>

#include "NorbLoader.h"
//#include "BoardHelper.h"
#include "Timer.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "stringhelper.h"
#include "FileHelper.h"
#include "StatefulTimer.h"
#include "WeightsPersister.h"
#include "NormalizationHelper.h"

using namespace std;

/* [[[cog
    # These are used in the later cog sections in this file:
    strings = [ 'dataDir', 'trainSet', 'testSet', 'netDef', 'restartableFilename', 'normalization' ]
    ints = [ 'numTrain', 'numTest', 'batchSize', 'numEpochs', 'restartable' ]
    floats = [ 'learningRate', 'annealLearningRate' ]
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
        'restartablefilename': 'filename to store weights',
        'normalization': '[stddev|maxmin]'
    }
*///]]]
// [[[end]]]

class Config {
public:
    // [[[cog
    // cog.outl('// generated using cog:')
    // for astring in strings:
    //    cog.outl( 'string ' + astring + ' = "";')
    // for anint in ints:
    //    cog.outl( 'int ' + anint + ' = 0;')
    // for name in floats:
    //    cog.outl( 'float ' + name + ' = 0.0f;')
    // ]]]
    // generated using cog:
    string dataDir = "";
    string trainSet = "";
    string testSet = "";
    string netDef = "";
    string restartableFilename = "";
    string normalization = "";
    int numTrain = 0;
    int numTest = 0;
    int batchSize = 0;
    int numEpochs = 0;
    int restartable = 0;
    float learningRate = 0.0f;
    float annealLearningRate = 0.0f;
    // [[[end]]]

    Config() {
        netDef = "8C5-MP2-16C5-MP3-10N";
        dataDir = "../data/mnist";
        testSet = "t10k";
        restartableFilename = "weights.dat";
        normalization = "stddev";
        trainSet = "train";
        batchSize = 128;
        numTest = 0;
        restartable = 0;
        numTrain = 0;
        numEpochs = 12;
        learningRate = 0.002f;
        annealLearningRate = 1.0f;
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
    if( config.normalization == "stddev" ) {
        NormalizationHelper::getMeanAndStdDev( trainData, Ntrain * inputCubeSize, &mean, &stdDev );
    } else if( config.normalization == "maxmin" ) {
        NormalizationHelper::getMinMax( trainData, Ntrain * inputCubeSize, &mean, &stdDev );
    } else {
        cout << "Error: Unknown normalization: " << config.normalization << endl;
        return;
    }
    cout << " board stats mean " << mean << " stdDev " << stdDev << endl;
    timer.timeCheck("after getting stats");

    const int numToTrain = Ntrain;
    const int batchSize = config.batchSize;
    NeuralNet *net = NeuralNet::maker()->planes(numPlanes)->boardSize(boardSize)->instance();

    string netDefLower = toLower( config.netDef );
    vector<string> splitNetDef = split( netDefLower, "-" );
    for( int i = 0; i < splitNetDef.size(); i++ ) {
        string thisLayerDef = splitNetDef[i];
        vector<string>splitLayerDef = split( thisLayerDef, "{" );
        string baseLayerDef = splitLayerDef[0];
        string optionsDef = "";
        vector<string> splitOptionsDef;
        if( splitLayerDef.size() == 2 ) {
            optionsDef = split( splitLayerDef[1], "}" )[0];
            splitOptionsDef = split( optionsDef, "," );
        }
        cout << "optionsDef: " << optionsDef << endl;
        if( baseLayerDef.find("c") != string::npos ) {
            vector<string> splitConvDef = split( baseLayerDef, "c" );
            int numFilters = atoi( splitConvDef[0] );
            int filterSize = atoi( splitConvDef[1] );
            int skip = 0;
            ActivationFunction *fn = new ReluActivation();
            for( int i = 0; i < splitOptionsDef.size(); i++ ) {
                string optionDef = splitOptionsDef[i];
                cout << "optionDef: " << optionDef << endl;
                vector<string> splitOptionDef = split( optionDef, "=");
                string optionName = splitOptionDef[0];
                if( splitOptionDef.size() == 2 ) {
                    string optionValue = splitOptionDef[1];
                    if( optionName == "skip" ) {
                        skip = atoi( optionValue );
                        cout << "got skip: " << skip << endl;
                    }
                } else if( splitOptionDef.size() == 1 ) {
                    if( optionName == "tanh" ) {
                        fn = new TanhActivation();
                    } else if( optionName == "scaledtanh" ) {
                        fn = new ScaledTanhActivation();
                    } else if( optionName == "sigmoid" ) {
                        fn = new SigmoidActivation();
                    } else if( optionName == "linear" ) {
                        fn = new LinearActivation();
                    } else {
                        cout << "Error: unknown subkey: [" << splitOptionsDef[i] << "]" << endl;
                        return;
                    }
                } else {
                    cout << "Error: unknown subkey: [" << splitOptionsDef[i] << "]" << endl;
                    return;
                }
            }
            net->convolutionalMaker()->numFilters(numFilters)->filterSize(filterSize)->fn( fn )->biased()->insert();
        } else if( baseLayerDef.find("mp") != string::npos ) {
            vector<string> splitPoolDef = split( baseLayerDef, "mp" );
            int poolingSize = atoi( splitPoolDef[1] );
            net->poolingMaker()->poolingSize(poolingSize)->insert();
        } else if( baseLayerDef.find("n") != string::npos ) {
            vector<string> fullDef = split( baseLayerDef, "n" );
            int numPlanes = atoi( fullDef[0] );
            if( i == splitNetDef.size() - 1 ) {
                net->fullyConnectedMaker()->numPlanes(numPlanes)->boardSize(1)->linear()->biased()->insert();
            } else {
                net->fullyConnectedMaker()->numPlanes(numPlanes)->boardSize(1)->tanh()->biased()->insert();
            }
        } else {
            cout << "network definition " << baseLayerDef << " not recognised" << endl;
            return;
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
    //float annealedLearningRate = afterRestart ? restartAnnealedLearningRate : config.learningRate;
    for( int epoch = afterRestart ? restartEpoch : 0; epoch < config.numEpochs; epoch++ ) {
        float annealedLearningRate = config.learningRate * pow( config.annealLearningRate, epoch );
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
//        annealedLearningRate *= config.annealLearningRate;
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
    // for name in strings:
    //    cog.outl( 'cout << "    ' + name.lower() + '=[' + descriptions[name.lower()] + '] (" << config.' + name + ' << ")" << endl;')
    // for name in ints:
    //    cog.outl( 'cout << "    ' + name.lower() + '=[' + descriptions[name.lower()] + '] (" << config.' + name + ' << ")" << endl;')
    // for name in floats:
    //    cog.outl( 'cout << "    ' + name.lower() + '=[' + descriptions[name.lower()] + '] (" << config.' + name + ' << ")" << endl;')
    // ]]]
    // generated using cog:
    cout << "    datadir=[data directory] (" << config.dataDir << ")" << endl;
    cout << "    trainset=[[training-shuffled|testing-sampled|other set name]] (" << config.trainSet << ")" << endl;
    cout << "    testset=[[training-shuffled|testing-sampled|other set name]] (" << config.testSet << ")" << endl;
    cout << "    netdef=[network definition] (" << config.netDef << ")" << endl;
    cout << "    restartablefilename=[filename to store weights] (" << config.restartableFilename << ")" << endl;
    cout << "    normalization=[[stddev|maxmin]] (" << config.normalization << ")" << endl;
    cout << "    numtrain=[num training examples] (" << config.numTrain << ")" << endl;
    cout << "    numtest=[num test examples]] (" << config.numTest << ")" << endl;
    cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
    cout << "    numepochs=[number epochs] (" << config.numEpochs << ")" << endl;
    cout << "    restartable=[weights are persistent?] (" << config.restartable << ")" << endl;
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
            // for name in strings:
            //    cog.outl( '} else if( key == "' + name.lower() + '" ) {')
            //    cog.outl( '    config.' + name + ' = value;')
            // for name in ints:
            //    cog.outl( '} else if( key == "' + name.lower() + '" ) {')
            //    cog.outl( '    config.' + name + ' = atoi( value );')
            // for name in floats:
            //    cog.outl( '} else if( key == "' + name.lower() + '" ) {')
            //    cog.outl( '    config.' + name + ' = atof( value );')
            // ]]]
            // generated using cog:
            if( false ) {
            } else if( key == "datadir" ) {
                config.dataDir = value;
            } else if( key == "trainset" ) {
                config.trainSet = value;
            } else if( key == "testset" ) {
                config.testSet = value;
            } else if( key == "netdef" ) {
                config.netDef = value;
            } else if( key == "restartablefilename" ) {
                config.restartableFilename = value;
            } else if( key == "normalization" ) {
                config.normalization = value;
            } else if( key == "numtrain" ) {
                config.numTrain = atoi( value );
            } else if( key == "numtest" ) {
                config.numTest = atoi( value );
            } else if( key == "batchsize" ) {
                config.batchSize = atoi( value );
            } else if( key == "numepochs" ) {
                config.numEpochs = atoi( value );
            } else if( key == "restartable" ) {
                config.restartable = atoi( value );
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
    try {
        go( config );
    } catch( runtime_error e ) {
        cout << "Something went wrong: " << e.what() << endl;
        return -1;
    }
}


