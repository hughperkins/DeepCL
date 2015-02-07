// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>

//#include "NorbLoader.h"
#include "GenericLoader.h"
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
#include "MultiNet.h"

using namespace std;

/* [[[cog
    # These are used in the later cog sections in this file:
    strings = [ 'trainFile', 'validateFile', 'netDef', 'restartableFilename', 'normalization' ]
    ints = [ 'numTrain', 'numTest', 'batchSize', 'numEpochs', 'restartable', 'dumpTimings', 'multiNet',
        'loadOnDemand' ]
    floats = [ 'learningRate', 'annealLearningRate', 'normalizationNumStds' ]
    descriptions = {
        'trainfile': 'path to training data file',
        'validatefile': 'path to validation data file',
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
        'normalizationnumstds': 'with stddev normalization, how many stddevs from mean is 1?',
        'dumptimings': 'dump detailed timings each epoch? [1|0]',
        'multinet': 'number of Mcdnn columns to train',
        'loadondemand': 'load data on demand [1|0]'
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
    string trainFile = "";
    string validateFile = "";
    string netDef = "";
    string restartableFilename = "";
    string normalization = "";
    int numTrain = 0;
    int numTest = 0;
    int batchSize = 0;
    int numEpochs = 0;
    int restartable = 0;
    int dumpTimings = 0;
    int multiNet = 0;
    int loadOnDemand = 0;
    float learningRate = 0.0f;
    float annealLearningRate = 0.0f;
    float normalizationNumStds = 0.0f;
    // [[[end]]]

    Config() {
        netDef = "8C5-MP2-16C5-MP3-10N";
//        dataDir = "../data/mnist";
        trainFile = "../data/mnist/train-dat.mat";
        validateFile = "../data/mnist/t10k-dat.mat";
        restartableFilename = "weights.dat";
        normalization = "stddev";
        batchSize = 128;
        numTest = 0;
        restartable = 0;
        numTrain = 0;
        numEpochs = 12;
        learningRate = 0.002f;
        annealLearningRate = 1.0f;
        normalizationNumStds = 2.0f;
        dumpTimings = 0;
        multiNet = 1;
    }
    string getTrainingString() {
        string configString = "";
        configString += "netDef=" + netDef + " trainFile=" + trainFile;
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

template< typename T >
class BatchAction {
public:
    T *data;
    int *labels;
    BatchAction( T *data, int *labels ) :
        data(data),
        labels(labels) { // have to provide appropriate buffers for this
    }
    virtual void processBatch( int batchSize ) = 0;
};

template< typename T >
class NormalizeGetStdDev : public BatchAction<T> {
public:
    Statistics<T> statistics; 
    NormalizeGetStdDev( T *data, int *labels ) :
        BatchAction<T>::BatchAction( data, labels ) {
    }
    virtual void processBatch( int batchSize ) {
        NormalizationHelper::updateStatistics( this->data, batchSize, &statistics );
    }
    void calcMeanStdDev( float *p_mean, float *p_stdDev ) {
        NormalizationHelper::calcMeanAndStdDev( &statistics, p_mean, p_stdDev );
    }
};

template< typename T >
class NormalizeGetMinMax : public BatchAction<T> {
public:
    Statistics<T> statistics; 
    NormalizeGetMinMax( T *data, int *labels ) :
        BatchAction<T>( data, labels ) {
    }
    virtual void processBatch( int batchSize ) {
        NormalizationHelper::updateStatistics( this->data, batchSize, &statistics );
    }
    void calcMinMaxTransform( float *p_translate, float *p_scale ) {
        // add this to our values to center
        *p_translate = - ( statistics.maxY - statistics.minY ) / 2.0f;
        // multiply our values by this to scale to -1 / +1 range
        *p_scale = 1.0f / ( statistics.maxY - statistics.minY );
    }
};

class BatchProcess {
public:
    template< typename T>
    static void run(std::string filepath, int startN, int batchSize, int totalN, BatchAction<T> *batchAction) {
        int numBatches = ( totalN + batchSize - 1 ) / batchSize;
        int thisBatchSize = batchSize;
        for( int batch = 0; batch < numBatches; batch++ ) {
            int batchStart = batch * batchSize;
            if( batch == numBatches - 1 ) {
                thisBatchSize = totalN - batchStart;
                cout << "size of last batch: " << thisBatchSize << endl;
            }
            GenericLoader::load( filepath, batchAction->data, batchAction->labels, batchStart, thisBatchSize );
            batchAction->processBatch( thisBatchSize );
        }
    }
};

//void calcMeanStdDev( std::string filepath, int N, float *p_mean, float *p_stdDev ) {
//}

void go(Config config) {
    Timer timer;

    int Ntrain;
    int Ntest;
    int numPlanes;
    int boardSize;

    unsigned char *trainData = 0;
    unsigned char *testData = 0;
    int *trainLabels = 0;
    int *testLabels = 0;

    int trainAllocateN = 0;
    int testAllocateN = 0;

    int totalLinearSize;
    GenericLoader::getDimensions( config.trainFile, &Ntrain, &numPlanes, &boardSize, &totalLinearSize );
    Ntrain = config.numTrain == 0 ? Ntrain : config.numTrain;
//    long allocateSize = (long)Ntrain * numPlanes * boardSize * boardSize;
    cout << "Ntrain " << Ntrain << " numPlanes " << numPlanes << " boardSize " << boardSize << endl;
    if( config.loadOnDemand ) {
        trainAllocateN = config.batchSize; // can improve this later
    } else {
        trainAllocateN = Ntrain;
    }
    trainData = new unsigned char[ (long)trainAllocateN * numPlanes * boardSize * boardSize ];
    trainLabels = new int[trainAllocateN];
    if( !config.loadOnDemand ) {
        GenericLoader::load( config.trainFile, trainData, trainLabels, 0, Ntrain );
    }

    GenericLoader::getDimensions( config.validateFile, &Ntest, &numPlanes, &boardSize, &totalLinearSize );
    Ntest = config.numTest == 0 ? Ntest : config.numTest;
    if( config.loadOnDemand ) {
        testAllocateN = config.batchSize; // can improve this later
    } else {
        testAllocateN = Ntest;
    }
    testData = new unsigned char[ (long)testAllocateN * numPlanes * boardSize * boardSize ];
    testLabels = new int[testAllocateN];    
    if( !config.loadOnDemand ) {
        GenericLoader::load( config.validateFile, testData, testLabels, 0, Ntest );
    }
    
    timer.timeCheck("after load images");

    const int inputCubeSize = numPlanes * boardSize * boardSize;
    float translate;
    float scale;
    if( !config.loadOnDemand ) {
        if( config.normalization == "stddev" ) {
            float mean, stdDev;
            NormalizationHelper::getMeanAndStdDev( trainData, Ntrain * inputCubeSize, &mean, &stdDev );
            cout << " board stats mean " << mean << " stdDev " << stdDev << endl;
            translate = - mean;
            scale = 1.0f / stdDev / config.normalizationNumStds;
        } else if( config.normalization == "maxmin" ) {
            float mean, stdDev;
            NormalizationHelper::getMinMax( trainData, Ntrain * inputCubeSize, &mean, &stdDev );
            translate = - mean;
            scale = 1.0f / stdDev;
        } else {
            cout << "Error: Unknown normalization: " << config.normalization << endl;
            return;
        }
    } else {
        if( config.normalization == "stddev" ) {
            float mean, stdDev;
            NormalizeGetStdDev<unsigned char> normalizeGetStdDev( trainData, trainLabels ); 
            BatchProcess::run<unsigned char>( config.trainFile, 0, config.batchSize, Ntrain, &normalizeGetStdDev );
            normalizeGetStdDev.calcMeanStdDev( &mean, &stdDev );
            cout << " board stats mean " << mean << " stdDev " << stdDev << endl;
            translate = - mean;
            scale = 1.0f / stdDev / config.normalizationNumStds;
        } else if( config.normalization == "maxmin" ) {
            NormalizeGetMinMax<unsigned char> normalizeGetMinMax( trainData, trainLabels );
            BatchProcess::run( config.trainFile, 0, config.batchSize, Ntrain, &normalizeGetMinMax );
            normalizeGetMinMax.calcMinMaxTransform( &translate, &scale );
        } else {
            cout << "Error: Unknown normalization: " << config.normalization << endl;
            return;
        }
    }
    cout << " board norm translate " << translate << " scale " << scale << endl;
    timer.timeCheck("after getting stats");

    const int numToTrain = Ntrain;
    const int batchSize = config.batchSize;
    NeuralNet *net = new NeuralNet();
//    net->inputMaker<unsigned char>()->numPlanes(numPlanes)->boardSize(boardSize)->insert();
    net->addLayer( InputLayerMaker<unsigned char>::instance()->numPlanes(numPlanes)->boardSize(boardSize) );
    net->addLayer( NormalizationLayerMaker::instance()->translate(translate)->scale(scale) );
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

    Trainable *trainable = net;
    MultiNet *multiNet = 0;
    if( config.multiNet > 1 ) {
        multiNet = new MultiNet( config.multiNet, net );
        trainable = multiNet;
    }
//    } else {
    NetLearner<unsigned char> netLearner( trainable );
    netLearner.setTrainingData( Ntrain, trainData, trainLabels );
    netLearner.setTestingData( Ntest, testData, testLabels );
    netLearner.setSchedule( config.numEpochs, afterRestart ? restartEpoch : 1 );
    netLearner.setBatchSize( config.batchSize );
    netLearner.setDumpTimings( config.dumpTimings );
    WeightsWriter weightsWriter( net, &config );
    if( config.restartable ) {
        netLearner.addPostEpochAction( &weightsWriter );
    }
    netLearner.learn( config.learningRate, config.annealLearningRate );
//    }

    if( multiNet != 0 ) {
        delete multiNet;
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
    cout << "    trainfile=[path to training data file] (" << config.trainFile << ")" << endl;
    cout << "    validatefile=[path to validation data file] (" << config.validateFile << ")" << endl;
    cout << "    netdef=[network definition] (" << config.netDef << ")" << endl;
    cout << "    restartablefilename=[filename to store weights] (" << config.restartableFilename << ")" << endl;
    cout << "    normalization=[[stddev|maxmin]] (" << config.normalization << ")" << endl;
    cout << "    numtrain=[num training examples] (" << config.numTrain << ")" << endl;
    cout << "    numtest=[num test examples]] (" << config.numTest << ")" << endl;
    cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
    cout << "    numepochs=[number epochs] (" << config.numEpochs << ")" << endl;
    cout << "    restartable=[weights are persistent?] (" << config.restartable << ")" << endl;
    cout << "    dumptimings=[dump detailed timings each epoch? [1|0]] (" << config.dumpTimings << ")" << endl;
    cout << "    multinet=[number of Mcdnn columns to train] (" << config.multiNet << ")" << endl;
    cout << "    loadondemand=[load data on demand [1|0]] (" << config.loadOnDemand << ")" << endl;
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
            } else if( key == "trainfile" ) {
                config.trainFile = value;
            } else if( key == "validatefile" ) {
                config.validateFile = value;
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
            } else if( key == "dumptimings" ) {
                config.dumpTimings = atoi( value );
            } else if( key == "multinet" ) {
                config.multiNet = atoi( value );
            } else if( key == "loadondemand" ) {
                config.loadOnDemand = atoi( value );
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


