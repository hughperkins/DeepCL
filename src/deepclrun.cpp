// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>
#include <algorithm>

#include "GenericLoader.h"
#include "Timer.h"
#include "NeuralNet.h"
#include "stringhelper.h"
#include "FileHelper.h"
#include "StatefulTimer.h"
#include "WeightsPersister.h"
#include "NormalizationHelper.h"
//#include "BatchLearner.h"
#include "NetdefToNet.h"
#include "NetLearner.h"
#include "MultiNet.h"
#include "BatchProcess.h"
#include "NetLearnerOnDemand.h"

using namespace std;

/* [[[cog
    # These are used in the later cog sections in this file:
    # format:
    # ( name, type, description, default )
    options = [
        ('dataDir', 'string', 'directory to search for train and validate files', '../data/mnist' ),
        ('trainFile', 'string', 'path to training data file',"train-images-idx3-ubyte"),
        ('dataset', 'string', 'choose datadir,trainfile,and validatefile for certain datasets [mnist|norb|kgsgo|cifar10]',''),
        ('validateFile', 'string', 'path to validation data file',"t10k-images-idx3-ubyte"),
        ('numTrain', 'int', 'num training examples',-1),
        ('numTest', 'int', 'num test examples]',-1),
        ('batchSize', 'int', 'batch size',128),
        ('numEpochs', 'int', 'number epochs',12),
        ('netDef', 'string', 'network definition',"RT2-8C5{z}-MP2-16C5{z}-MP3-150N-10N"),
        ('learningRate', 'float', 'learning rate, a float value', 0.002),
        ('annealLearningRate', 'float', 'multiply learning rate by this, each epoch',1),
        ('loadWeights', 'int', 'load weights from file at startup?', 0),
        ('weightsFile', 'string', 'file to write weights to','weights.dat'),
        ('writeWeightsInterval', 'float', 'write weights every this many minutes', 0 ),
        ('normalization', 'string', '[stddev|maxmin]', 'stddev'),
        ('normalizationNumStds', 'float', 'with stddev normalization, how many stddevs from mean is 1?', 2.0),
        ('dumpTimings', 'int', 'dump detailed timings each epoch? [1|0]', 0),
        ('multiNet', 'int', 'number of Mcdnn columns to train', 1),
        ('loadOnDemand', 'int', 'load data on demand [1|0]', 0),
        ('fileReadBatches', 'int', 'how many batches to read from file each time? (for loadondemand=1)', 50),
        ('normalizationExamples', 'int', 'number of examples to read to determine normalization parameters', 10000)
    ]
*///]]]
// [[[end]]]

class Config {
public:
    /* [[[cog
        cog.outl('// generated using cog:')
        for (name,type,description,default) in options:
            cog.outl( type + ' ' + name + ';')
    */// ]]]
    // generated using cog:
    string dataDir;
    string trainFile;
    string dataset;
    string validateFile;
    int numTrain;
    int numTest;
    int batchSize;
    int numEpochs;
    string netDef;
    float learningRate;
    float annealLearningRate;
    int loadWeights;
    string weightsFile;
    float writeWeightsInterval;
    string normalization;
    float normalizationNumStds;
    int dumpTimings;
    int multiNet;
    int loadOnDemand;
    int fileReadBatches;
    int normalizationExamples;
    // [[[end]]]

    Config() {
        /* [[[cog
            cog.outl('// generated using cog:')
            for (name,type,description,default) in options:
                # initializer = ''
                # if type == 'string':
                #     initializer = '""'
                # elif type == 'int':
                #     initializer = '0'
                # else:
                #     initializer = '0.0f'
                defaultString = ''
                if type == 'string':
                    defaultString = '"' + default + '"'
                elif type == 'int':
                    defaultString = str(default)
                elif type == 'float':
                    defaultString = str(default)
                    if '.' not in defaultString:
                        defaultString += '.0'
                    defaultString += 'f'
                cog.outl( name + ' = ' + defaultString + ';')
        */// ]]]
        // generated using cog:
        dataDir = "../data/mnist";
        trainFile = "train-images-idx3-ubyte";
        dataset = "";
        validateFile = "t10k-images-idx3-ubyte";
        numTrain = -1;
        numTest = -1;
        batchSize = 128;
        numEpochs = 12;
        netDef = "RT2-8C5{z}-MP2-16C5{z}-MP3-150N-10N";
        learningRate = 0.002f;
        annealLearningRate = 1.0f;
        loadWeights = 0;
        weightsFile = "weights.dat";
        writeWeightsInterval = 0.0f;
        normalization = "stddev";
        normalizationNumStds = 2.0f;
        dumpTimings = 0;
        multiNet = 1;
        loadOnDemand = 0;
        fileReadBatches = 50;
        normalizationExamples = 10000;
        // [[[end]]]
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
        WeightsPersister::persistWeights( config->weightsFile, config->getTrainingString(), net, epoch + 1, 0, 0, 0, 0 );
    }
};

class IntervalWeightsWriter : public NetLearner_PostBatchAction {
public:
    NeuralNet *net;
    Config *config;
    int intervalMinutes;
    double lastTime;
    IntervalWeightsWriter( NeuralNet *net, Config *config, int intervalMinutes ) :
            net( net ),
            config( config ),
            intervalMinutes( intervalMinutes ) {
        lastTime = 0;
    }
    virtual void run( int epoch, int batch, float loss, int numRight ) {
        cout << "intervalweightswriter" << endl;
        //WeightsPersister::persistWeights( config->weightsFile, config->getTrainingString(), net, epoch + 1, 0, 0, 0, 0 );
    }
};

void go(Config config) {
    Timer timer;

    int Ntrain;
    int Ntest;
    int numPlanes;
    int imageSize;

    unsigned char *trainData = 0;
    unsigned char *testData = 0;
    int *trainLabels = 0;
    int *testLabels = 0;

    int trainAllocateN = 0;
    int testAllocateN = 0;

//    int totalLinearSize;
    GenericLoader::getDimensions( config.dataDir + "/" + config.trainFile, &Ntrain, &numPlanes, &imageSize );
    Ntrain = config.numTrain == -1 ? Ntrain : config.numTrain;
//    long allocateSize = (long)Ntrain * numPlanes * imageSize * imageSize;
    cout << "Ntrain " << Ntrain << " numPlanes " << numPlanes << " imageSize " << imageSize << endl;
    if( config.loadOnDemand ) {
        trainAllocateN = config.batchSize; // can improve this later
    } else {
        trainAllocateN = Ntrain;
    }
    trainData = new unsigned char[ (long)trainAllocateN * numPlanes * imageSize * imageSize ];
    trainLabels = new int[trainAllocateN];
    if( !config.loadOnDemand && Ntrain > 0 ) {
        GenericLoader::load( config.dataDir + "/" + config.trainFile, trainData, trainLabels, 0, Ntrain );
    }

    GenericLoader::getDimensions( config.dataDir + "/" + config.validateFile, &Ntest, &numPlanes, &imageSize );
    Ntest = config.numTest == -1 ? Ntest : config.numTest;
    if( config.loadOnDemand ) {
        testAllocateN = config.batchSize; // can improve this later
    } else {
        testAllocateN = Ntest;
    }
    testData = new unsigned char[ (long)testAllocateN * numPlanes * imageSize * imageSize ];
    testLabels = new int[testAllocateN]; 
    if( !config.loadOnDemand && Ntest > 0 ) {
        GenericLoader::load( config.dataDir + "/" + config.validateFile, testData, testLabels, 0, Ntest );
    }
    cout << "Ntest " << Ntest << " Ntest" << endl;
    
    timer.timeCheck("after load images");

    const int inputCubeSize = numPlanes * imageSize * imageSize;
    float translate;
    float scale;
    int normalizationExamples = config.normalizationExamples > Ntrain ? Ntrain : config.normalizationExamples;
    if( !config.loadOnDemand ) {
        if( config.normalization == "stddev" ) {
            float mean, stdDev;
            NormalizationHelper::getMeanAndStdDev( trainData, normalizationExamples * inputCubeSize, &mean, &stdDev );
            cout << " image stats mean " << mean << " stdDev " << stdDev << endl;
            translate = - mean;
            scale = 1.0f / stdDev / config.normalizationNumStds;
        } else if( config.normalization == "maxmin" ) {
            float mean, stdDev;
            NormalizationHelper::getMinMax( trainData, normalizationExamples * inputCubeSize, &mean, &stdDev );
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
            BatchProcess::run<unsigned char>( config.dataDir + "/" + config.trainFile, 0, config.batchSize, normalizationExamples, inputCubeSize, &normalizeGetStdDev );
            normalizeGetStdDev.calcMeanStdDev( &mean, &stdDev );
            cout << " image stats mean " << mean << " stdDev " << stdDev << endl;
            translate = - mean;
            scale = 1.0f / stdDev / config.normalizationNumStds;
        } else if( config.normalization == "maxmin" ) {
            NormalizeGetMinMax<unsigned char> normalizeGetMinMax( trainData, trainLabels );
            BatchProcess::run( config.dataDir + "/" + config.trainFile, 0, config.batchSize, normalizationExamples, inputCubeSize, &normalizeGetMinMax );
            normalizeGetMinMax.calcMinMaxTransform( &translate, &scale );
        } else {
            cout << "Error: Unknown normalization: " << config.normalization << endl;
            return;
        }
    }
    cout << " image norm translate " << translate << " scale " << scale << endl;
    timer.timeCheck("after getting stats");

//    const int numToTrain = Ntrain;
//    const int batchSize = config.batchSize;
    NeuralNet *net = new NeuralNet();
//    net->inputMaker<unsigned char>()->numPlanes(numPlanes)->imageSize(imageSize)->insert();
    net->addLayer( InputLayerMaker<unsigned char>::instance()->numPlanes(numPlanes)->imageSize(imageSize) );
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
    if( config.loadWeights && config.weightsFile != "" ) {
        afterRestart = WeightsPersister::loadWeights( config.weightsFile, config.getTrainingString(), net, &restartEpoch, &restartBatch, &restartAnnealedLearningRate, &restartNumRight, &restartLoss );
        if( !afterRestart && FileHelper::exists( config.weightsFile ) ) {
            cout << "Weights file " << config.weightsFile << " exists, but doesnt match training options provided => aborting" << endl;
            cout << "Please either check the training options, or choose a weights file that doesnt exist yet" << endl;
            return;
        }
    }

    timer.timeCheck("before learning start");
    if( config.dumpTimings ) {
        StatefulTimer::dump( true );
    }
    StatefulTimer::timeCheck("START");

    Trainable *trainable = net;
    MultiNet *multiNet = 0;
    if( config.multiNet > 1 ) {
        multiNet = new MultiNet( config.multiNet, net );
        trainable = multiNet;
    }
    if( config.loadOnDemand ) {
        NetLearnerOnDemand<unsigned char> netLearner( trainable );
        netLearner.setTrainingData( config.dataDir + "/" + config.trainFile, Ntrain );
        netLearner.setTestingData( config.dataDir + "/" + config.validateFile, Ntest );
        netLearner.setSchedule( config.numEpochs, afterRestart ? restartEpoch : 1 );
        netLearner.setBatchSize( config.fileReadBatches, config.batchSize );
        netLearner.setDumpTimings( config.dumpTimings );
        WeightsWriter weightsWriter( net, &config );
        if( config.weightsFile != "" ) {
            netLearner.addPostEpochAction( &weightsWriter );
        }
        IntervalWeightsWriter intervalWeightsWriter( net, &config, config.writeWeightsInterval );
        if( config.writeWeightsInterval > 0 ) {
            netLearner.addPostBatchAction( &intervalWeightsWriter );
        }
        netLearner.learn( config.learningRate, config.annealLearningRate );
    } else {
        NetLearner<unsigned char> netLearner( trainable );
        netLearner.setTrainingData( Ntrain, trainData, trainLabels );
        netLearner.setTestingData( Ntest, testData, testLabels );
        netLearner.setSchedule( config.numEpochs, afterRestart ? restartEpoch : 1 );
        netLearner.setBatchSize( config.batchSize );
        netLearner.setDumpTimings( config.dumpTimings );
        WeightsWriter weightsWriter( net, &config );
        if( config.weightsFile != "" ) {
            netLearner.addPostEpochAction( &weightsWriter );
        }
        IntervalWeightsWriter intervalWeightsWriter( net, &config, config.writeWeightsInterval );
        if( config.writeWeightsInterval > 0 ) {
            netLearner.addPostBatchAction( &intervalWeightsWriter );
        }
        netLearner.learn( config.learningRate, config.annealLearningRate );
    }

    if( multiNet != 0 ) {
        delete multiNet;
    }
    delete net;

    if( trainData != 0 ) {
        delete[] trainData;
    }
    if( testData != 0 ) {
        delete[] testData;
    }
    if( testLabels != 0 ) {
        delete[] testLabels;
    }
    if( trainLabels != 0 ) {
        delete[] trainLabels;
    }
}

void printUsage( char *argv[], Config config ) {
    cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
    cout << endl;
    cout << "Possible key=value pairs:" << endl;
    /* [[[cog
        cog.outl('// generated using cog:')
        for (name,type,description,_) in options:
            cog.outl( 'cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
    *///]]]
    // generated using cog:
    cout << "    datadir=[directory to search for train and validate files] (" << config.dataDir << ")" << endl;
    cout << "    trainfile=[path to training data file] (" << config.trainFile << ")" << endl;
    cout << "    dataset=[choose datadir,trainfile,and validatefile for certain datasets [mnist|norb|kgsgo|cifar10]] (" << config.dataset << ")" << endl;
    cout << "    validatefile=[path to validation data file] (" << config.validateFile << ")" << endl;
    cout << "    numtrain=[num training examples] (" << config.numTrain << ")" << endl;
    cout << "    numtest=[num test examples]] (" << config.numTest << ")" << endl;
    cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
    cout << "    numepochs=[number epochs] (" << config.numEpochs << ")" << endl;
    cout << "    netdef=[network definition] (" << config.netDef << ")" << endl;
    cout << "    learningrate=[learning rate, a float value] (" << config.learningRate << ")" << endl;
    cout << "    anneallearningrate=[multiply learning rate by this, each epoch] (" << config.annealLearningRate << ")" << endl;
    cout << "    loadweights=[load weights from file at startup?] (" << config.loadWeights << ")" << endl;
    cout << "    weightsfile=[file to write weights to] (" << config.weightsFile << ")" << endl;
    cout << "    writeweightsinterval=[write weights every this many minutes] (" << config.writeWeightsInterval << ")" << endl;
    cout << "    normalization=[[stddev|maxmin]] (" << config.normalization << ")" << endl;
    cout << "    normalizationnumstds=[with stddev normalization, how many stddevs from mean is 1?] (" << config.normalizationNumStds << ")" << endl;
    cout << "    dumptimings=[dump detailed timings each epoch? [1|0]] (" << config.dumpTimings << ")" << endl;
    cout << "    multinet=[number of Mcdnn columns to train] (" << config.multiNet << ")" << endl;
    cout << "    loadondemand=[load data on demand [1|0]] (" << config.loadOnDemand << ")" << endl;
    cout << "    filereadbatches=[how many batches to read from file each time? (for loadondemand=1)] (" << config.fileReadBatches << ")" << endl;
    cout << "    normalizationexamples=[number of examples to read to determine normalization parameters] (" << config.normalizationExamples << ")" << endl;
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
//            cout << "key [" << key << "]" << endl;
            /* [[[cog
                cog.outl('// generated using cog:')
                cog.outl('if( false ) {')
                for (name,type,description,_) in options:
                    cog.outl( '} else if( key == "' + name.lower() + '" ) {')
                    converter = '';
                    if type == 'int':
                        converter = 'atoi';
                    elif type == 'float':
                        converter = 'atof';
                    cog.outl( '    config.' + name + ' = ' + converter + '(value);')
            */// ]]]
            // generated using cog:
            if( false ) {
            } else if( key == "datadir" ) {
                config.dataDir = (value);
            } else if( key == "trainfile" ) {
                config.trainFile = (value);
            } else if( key == "dataset" ) {
                config.dataset = (value);
            } else if( key == "validatefile" ) {
                config.validateFile = (value);
            } else if( key == "numtrain" ) {
                config.numTrain = atoi(value);
            } else if( key == "numtest" ) {
                config.numTest = atoi(value);
            } else if( key == "batchsize" ) {
                config.batchSize = atoi(value);
            } else if( key == "numepochs" ) {
                config.numEpochs = atoi(value);
            } else if( key == "netdef" ) {
                config.netDef = (value);
            } else if( key == "learningrate" ) {
                config.learningRate = atof(value);
            } else if( key == "anneallearningrate" ) {
                config.annealLearningRate = atof(value);
            } else if( key == "loadweights" ) {
                config.loadWeights = atoi(value);
            } else if( key == "weightsfile" ) {
                config.weightsFile = (value);
            } else if( key == "writeweightsinterval" ) {
                config.writeWeightsInterval = atof(value);
            } else if( key == "normalization" ) {
                config.normalization = (value);
            } else if( key == "normalizationnumstds" ) {
                config.normalizationNumStds = atof(value);
            } else if( key == "dumptimings" ) {
                config.dumpTimings = atoi(value);
            } else if( key == "multinet" ) {
                config.multiNet = atoi(value);
            } else if( key == "loadondemand" ) {
                config.loadOnDemand = atoi(value);
            } else if( key == "filereadbatches" ) {
                config.fileReadBatches = atoi(value);
            } else if( key == "normalizationexamples" ) {
                config.normalizationExamples = atoi(value);
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
    string dataset = toLower( config.dataset );
    if( dataset != "" ) {
        if( dataset == "mnist" ) {
            config.dataDir = "../data/mnist";
            config.trainFile = "train-images-idx3-ubyte";
            config.validateFile = "t10k-images-idx3-ubyte";
        } else if( dataset == "norb" ) {
            config.dataDir = "../data/norb";
            config.trainFile = "training-shuffled-dat.mat";
            config.validateFile = "testing-sampled-dat.mat";
        } else if( dataset == "cifar10" ) {
            config.dataDir = "../data/cifar10";
            config.trainFile = "train-dat.mat";
            config.validateFile = "test-dat.mat";
        } else if( dataset == "kgsgo" ) {
            config.dataDir = "../data/kgsgo";
            config.trainFile = "kgsgo-train10k-v2.dat";
            config.validateFile = "kgsgo-test-v2.dat";
            config.loadOnDemand = 1;
        } else if( dataset == "kgsgoall" ) {
            config.dataDir = "../data/kgsgo";
            config.trainFile = "kgsgo-trainall-v2.dat";
            config.validateFile = "kgsgo-test-v2.dat";
            config.loadOnDemand = 1;
        } else {
            cout << "dataset " << dataset << " not known.  please choose from: mnist, norb, cifar10, kgsgo" << endl;
            return -1;
        }
        cout << "Using dataset " << dataset << ":" << endl;
        cout << "   datadir: " << config.dataDir << ":" << endl;
        cout << "   trainfile: " << config.trainFile << ":" << endl;
        cout << "   validatefile: " << config.validateFile << ":" << endl;
    }
    try {
        go( config );
    } catch( runtime_error e ) {
        cout << "Something went wrong: " << e.what() << endl;
        return -1;
    }
}


