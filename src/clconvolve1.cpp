// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>

#include "NorbLoader.h"
#include "Timer.h"
#include "NeuralNet.h"
#include "stringhelper.h"
#include "FileHelper.h"
#include "StatefulTimer.h"
#include "WeightsPersister.h"
#include "NormalizationHelper.h"
#include "BatchLearner.h"
#include "NetdefToNet.h"
#include "NetLearner.h"

using namespace std;

/* [[[cog
    # These are used in the later cog sections in this file:
    strings = [ 'dataDir', 'trainSet', 'testSet', 'netDef', 'restartableFilename', 'normalization' ]
    ints = [ 'numTrain', 'numTest', 'batchSize', 'numEpochs', 'restartable' ]
    floats = [ 'learningRate', 'annealLearningRate', 'normalizationNumStds' ]
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
        'normalization': '[stddev|maxmin]',
        'normalizationnumstds': 'with stddev normalization, how many stddevs from mean is 1?'
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
    float normalizationNumStds = 0.0f;
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
        normalizationNumStds = 2.0f;
    }
    string getTrainingString() {
        string configString = "";
        configString += "netDef=" + netDef + " dataDir=" + dataDir + " trainSet=" + trainSet;
        return configString;
    }
};

class WeightsWriter : public PostEpochAction {
public:
    NeuralNet *net;
    Config *config;
    WeightsWriter( NeuralNet *net, Config *config ) :
        net( net ),
        config( config ) {
    }
    virtual void run( int epoch ) {
        WeightsPersister::persistWeights( config->restartableFilename, config->getTrainingString(), net, epoch + 1, 0, 0, 0, 0 );
    }
};

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
        stdDev *= config.normalizationNumStds;
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
    if( !NetdefToNet::createNetFromNetdef( net, config.netDef ) ) {
        return;
    }
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

    NetLearner<unsigned char> netLearner( net );
    netLearner.setTrainingData( Ntrain, trainData, trainLabels );
    netLearner.setTestingData( Ntest, testData, testLabels );
    netLearner.setSchedule( config.numEpochs, afterRestart ? restartEpoch : 1 );
    netLearner.setNormalize( - mean, 1.0f / stdDev );
    netLearner.setBatchSize( config.batchSize );
    WeightsWriter weightsWriter( net, &config );
    if( config.restartable ) {
        netLearner.addPostEpochAction( &weightsWriter );
    }
    netLearner.learn( config.learningRate, config.annealLearningRate );

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
    cout << "    normalizationnumstds=[with stddev normalization, how many stddevs from mean is 1?] (" << config.normalizationNumStds << ")" << endl;
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
            } else if( key == "normalizationnumstds" ) {
                config.normalizationNumStds = atof( value );
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


