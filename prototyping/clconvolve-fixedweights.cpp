// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// this initializes the weights using a fixed seed, repeatably
// to enable comparison with convnetjs


#include <iostream>
#include <algorithm>
#include <random>

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
#include "FullyConnectedLayer.h"
#include "PoolingLayer.h"
#include "SoftMaxLayer.h"

using namespace std;

/* [[[cog
    # These are used in the later cog sections in this file:
    strings = [ 'dataDir', 'trainFile', 'validateFile', 'netDef', 'normalization', 'dataset' ]
    ints = [ 'numTrain', 'numTest', 'batchSize', 'numEpochs', 'dumpTimings', 
        'normalizationExamples' ]
    floats = [ 'learningRate', 'normalizationNumStds' ]
    descriptions = {
        'dataset': 'choose datadir,trainfile,and validatefile for certain datasets [mnist|norb|kgsgo|cifar10]',
        'datadir': 'directory to search for train and validate files',
        'trainfile': 'path to training data file',
        'validatefile': 'path to validation data file',
        'numtrain': 'num training examples',
        'numtest': 'num test examples]',
        'batchsize': 'batch size',
        'numepochs': 'number epochs',
        'netdef': 'network definition',
        'learningrate': 'learning rate, a float value',
        'anneallearningrate': 'multiply learning rate by this, each epoch',
        'loadweights': 'weights are persistent?',
        'weightsfile': 'load weights from weights file at startup?',
        'normalization': '[stddev|maxmin]',
        'normalizationnumstds': 'with stddev normalization, how many stddevs from mean is 1?',
        'dumptimings': 'dump detailed timings each epoch? [1|0]',
        'multinet': 'number of Mcdnn columns to train',
        'loadondemand': 'load data on demand [1|0]',
        'filereadbatches': 'how many batches to read from file each time? (for loadondemand=1)',
        'normalizationexamples': 'number of examples to read to determine normalization parameters'
    }
*///]]]
// [[[end]]]

mt19937 initrand;

class Config {
public:
    // [[[cog
    // cog.outl('// generated using cog:')
    // for astring in strings:
    //    cog.outl( 'string ' + astring + ';')
    // for anint in ints:
    //    cog.outl( 'int ' + anint + ';')
    // for name in floats:
    //    cog.outl( 'float ' + name + ';')
    // ]]]
    // generated using cog:
    string dataDir;
    string trainFile;
    string validateFile;
    string netDef;
    string normalization;
    string dataset;
    int numTrain;
    int numTest;
    int batchSize;
    int numEpochs;
    int dumpTimings;
    int normalizationExamples;
    float learningRate;
    float normalizationNumStds;
    // [[[end]]]

    Config() {
        // [[[cog
        // cog.outl('// generated using cog:')
        // for astring in strings:
        //    cog.outl( astring + ' = "";')
        // for anint in ints:
        //    cog.outl( anint + ' = 0;')
        // for name in floats:
        //    cog.outl( name + ' = 0.0f;')
        // ]]]
        // generated using cog:
        dataDir = "";
        trainFile = "";
        validateFile = "";
        netDef = "";
        normalization = "";
        dataset = "";
        numTrain = 0;
        numTest = 0;
        batchSize = 0;
        numEpochs = 0;
        dumpTimings = 0;
        normalizationExamples = 0;
        learningRate = 0.0f;
        normalizationNumStds = 0.0f;
    // [[[end]]]
        netDef = "10N{linear}";
//        dataDir = "../data/mnist";
        dataDir = "../data/mnist";
        trainFile = "train-dat.mat";
        validateFile = "t10k-dat.mat";
        normalization = "stddev";
        batchSize = 128;
        numTest = 0;
        numTrain = -1;
        numEpochs = 12;
        normalizationExamples = 1;
        learningRate = 0.002f;
        normalizationNumStds = 2.0f;
        dumpTimings = 0;
    }
    string getTrainingString() {
        string configString = "";
        configString += "netDef=" + netDef + " trainFile=" + trainFile;
        return configString;
    }
};

void sampleWeights( NeuralNet *net ) {
    for( int layerId = 0; layerId < (int)net->layers.size();  layerId++ ) {
        Layer *layer = net->layers[layerId];
        FullyConnectedLayer *fc = dynamic_cast< FullyConnectedLayer * >( layer );
        ConvolutionalLayer *conv = dynamic_cast< ConvolutionalLayer * >( layer );
        if( fc != 0 ) {
            conv = fc->convolutionalLayer;
        }
        if( conv == 0 ) {
            continue;
        }

        cout << "layer " << layerId << endl;
        float const*weights = conv->getWeights();
        conv->getBiasWeights();
        LayerDimensions &dim = conv->dim;
        int numFilters = dim.numFilters;
        int inputPlanes = dim.inputPlanes;
        int filterSize = dim.filterSize;

        initrand.seed(0);        
        for( int i = 0; i < 10; i++ ) {
            int thisrand = abs( (int)initrand() );
            int seq = thisrand % ( numFilters * inputPlanes * filterSize * filterSize );
            int planefilter = seq / ( filterSize * filterSize );
            int rowcol = seq % ( filterSize * filterSize );
            int filter = planefilter / inputPlanes;
            int inputPlane = planefilter % inputPlanes;
            int row = rowcol / filterSize;
            int col = rowcol % filterSize;
            cout << "weights[" << filter << "," << inputPlane << "," << row << "," << col << "]=" << weights[ seq ] << endl;
        }
    }
}

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
    trainAllocateN = Ntrain;
    trainData = new unsigned char[ (long)trainAllocateN * numPlanes * imageSize * imageSize ];
    trainLabels = new int[trainAllocateN];
    if( Ntrain > 0 ) {
        GenericLoader::load( config.dataDir + "/" + config.trainFile, trainData, trainLabels, 0, Ntrain );
    }

    GenericLoader::getDimensions( config.dataDir + "/" + config.validateFile, &Ntest, &numPlanes, &imageSize );
    Ntest = config.numTest == -1 ? Ntest : config.numTest;
    testAllocateN = Ntest;
    testData = new unsigned char[ (long)testAllocateN * numPlanes * imageSize * imageSize ];
    testLabels = new int[testAllocateN]; 
    if( Ntest > 0 ) {
        GenericLoader::load( config.dataDir + "/" + config.validateFile, testData, testLabels, 0, Ntest );
    }
    
    timer.timeCheck("after load images");

    const int inputCubeSize = numPlanes * imageSize * imageSize;
    float translate;
    float scale;
    int normalizationExamples = config.normalizationExamples > Ntrain ? Ntrain : config.normalizationExamples;
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
    for( int i = 1; i < (int)net->layers.size() - 1; i++ ) {
        Layer *layer = net->layers[i];
        FullyConnectedLayer *fc = dynamic_cast< FullyConnectedLayer * >(layer);
        ConvolutionalLayer *conv = dynamic_cast< ConvolutionalLayer * >(layer);
        if( fc != 0 ) {
            conv = fc->convolutionalLayer;
        }
        if( conv == 0 ) {
            continue;
        }
        initrand.seed(0);
        int weightsSize = conv->getWeightsSize();
    //int weightsSize = layer->getPersistSize();
        if( weightsSize > 0 ) {
            cout << "weightsSize " << weightsSize << endl;
            float *weights = new float[weightsSize];
            for( int j = 0; j < weightsSize; j++ ) {
                int thisrand = (int)initrand();
                float thisweight = ( thisrand % 100000 ) / 1000000.0f;
                weights[j] = thisweight;
            }        
            conv->initWeights( weights );
        }
        if( conv->dim.biased ) {
            initrand.seed(0);
            int biasedSize = conv->getBiasWeightsSize();
            float *biasWeights = new float[biasedSize];
            for( int j = 0; j < biasedSize; j++ ) {
                int thisrand = (int)initrand();
                float thisweight = ( thisrand % 100000 ) / 1000000.0f;
                biasWeights[j] = thisweight;
                //biasWeights[j] = 0;
            }        
            conv->initBiasWeights( biasWeights );
        }
    }

    cout << "weight samples before learning:" << endl;
    sampleWeights(net);

    bool afterRestart = false;
    int restartEpoch = 0;
//    int restartBatch = 0;
//    float restartAnnealedLearningRate = 0;
//    int restartNumRight = 0;
//    float restartLoss = 0;

    timer.timeCheck("before learning start");
    if( config.dumpTimings ) {
        StatefulTimer::dump( true );
    }
    StatefulTimer::timeCheck("START");

    Trainable *trainable = net;
    NetLearner<unsigned char> netLearner( trainable );
    netLearner.setTrainingData( Ntrain, trainData, trainLabels );
    netLearner.setTestingData( Ntest, testData, testLabels );
    netLearner.setSchedule( config.numEpochs, afterRestart ? restartEpoch : 1 );
    netLearner.setBatchSize( config.batchSize );
    netLearner.setDumpTimings( config.dumpTimings );  
    netLearner.learn( config.learningRate, 1.0f );

    cout << "forward results" << endl;
    for( int layerId = 0; layerId < (int)net->layers.size(); layerId++ ) {
        Layer *layer = net->layers[layerId];
        FullyConnectedLayer *fc = dynamic_cast< FullyConnectedLayer * >( layer );
        ConvolutionalLayer *conv = dynamic_cast< ConvolutionalLayer * >( layer );
        PoolingLayer *pool = dynamic_cast< PoolingLayer * >( layer );
        SoftMaxLayer *softMax = dynamic_cast< SoftMaxLayer * >( layer );
        if( fc != 0 ) {
            conv = fc->convolutionalLayer;
        }
        int planes = 0;
        int imageSize = 0;
        if( conv != 0 ) {
            cout << "convolutional (or conv based, ie fc)" << endl;
            planes = conv->dim.numFilters;
            imageSize = conv->dim.outputImageSize;
          //  continue;
        } else if( pool != 0 ) {
            cout << "pooling" << endl;
            planes = pool->numPlanes;
            imageSize = pool->outputImageSize;
        } else if( softMax != 0 ) {
            cout << "softmax" << endl;
            planes = softMax->numPlanes;
            imageSize = softMax->imageSize;
        } else {
            continue;
        }
        cout << "layer " << layerId << endl;
//        conv->getResults();
        float const*results = layer->getResults();
//        for( int i = 0; i < 3; i++ ) {
//            cout << conv->getResults()[i] << endl;
//        }
        initrand.seed(0);
//        LayerDimensions &dim = conv->dim;
        for( int i = 0; i < 10; i++ ) {
            int thisrand = abs( (int)initrand() );
            int seq = thisrand % ( planes * imageSize * imageSize );
            int outPlane = seq / ( imageSize * imageSize );
            int rowcol = seq % ( imageSize * imageSize );
            int row = rowcol / imageSize;
            int col = rowcol % imageSize;
            cout << "out[" << outPlane << "," << row << "," << col << "]=" << results[ seq ] << endl;
        }
    }

    cout << "weight samples after learning:" << endl;
    sampleWeights(net);

    cout << "backprop results" << endl;
    for( int layerId = net->layers.size() - 1; layerId >= 0; layerId-- ) {
        Layer *layer = net->layers[layerId];
        FullyConnectedLayer *fc = dynamic_cast< FullyConnectedLayer * >( layer );
        ConvolutionalLayer *conv = dynamic_cast< ConvolutionalLayer * >( layer );
        if( fc != 0 ) {
            conv = fc->convolutionalLayer;
        }
        if( conv == 0 ) {
            continue;
        }

        cout << "layer " << layerId << endl;
        float const*weights = conv->getWeights();
        float const*biases = conv->getBiasWeights();
        int weightsSize = conv->getWeightsSize() / conv->dim.numFilters;
        for( int i = 0; i < weightsSize; i++ ) {
            cout << " weight " << i << " " << weights[i] << endl;
        }
        for( int i = 0; i < 3; i++ ) {
            cout << " bias " << i << " " << biases[i] << endl;
        }
    }
    cout << "done" << endl;

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
    cout << "    datadir=[directory to search for train and validate files] (" << config.dataDir << ")" << endl;
    cout << "    trainfile=[path to training data file] (" << config.trainFile << ")" << endl;
    cout << "    validatefile=[path to validation data file] (" << config.validateFile << ")" << endl;
    cout << "    netdef=[network definition] (" << config.netDef << ")" << endl;
    cout << "    normalization=[[stddev|maxmin]] (" << config.normalization << ")" << endl;
    cout << "    dataset=[choose datadir,trainfile,and validatefile for certain datasets [mnist|norb|kgsgo|cifar10]] (" << config.dataset << ")" << endl;
    cout << "    numtrain=[num training examples] (" << config.numTrain << ")" << endl;
    cout << "    numtest=[num test examples]] (" << config.numTest << ")" << endl;
    cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
    cout << "    numepochs=[number epochs] (" << config.numEpochs << ")" << endl;
    cout << "    dumptimings=[dump detailed timings each epoch? [1|0]] (" << config.dumpTimings << ")" << endl;
    cout << "    normalizationexamples=[number of examples to read to determine normalization parameters] (" << config.normalizationExamples << ")" << endl;
    cout << "    learningrate=[learning rate, a float value] (" << config.learningRate << ")" << endl;
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
//            cout << "key [" << key << "]" << endl;
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
            } else if( key == "trainfile" ) {
                config.trainFile = value;
            } else if( key == "validatefile" ) {
                config.validateFile = value;
            } else if( key == "netdef" ) {
                config.netDef = value;
            } else if( key == "normalization" ) {
                config.normalization = value;
            } else if( key == "dataset" ) {
                config.dataset = value;
            } else if( key == "numtrain" ) {
                config.numTrain = atoi( value );
            } else if( key == "numtest" ) {
                config.numTest = atoi( value );
            } else if( key == "batchsize" ) {
                config.batchSize = atoi( value );
            } else if( key == "numepochs" ) {
                config.numEpochs = atoi( value );
            } else if( key == "dumptimings" ) {
                config.dumpTimings = atoi( value );
            } else if( key == "normalizationexamples" ) {
                config.normalizationExamples = atoi( value );
            } else if( key == "learningrate" ) {
                config.learningRate = atof( value );
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
    string dataset = toLower( config.dataset );
    if( dataset != "" ) {
        if( dataset == "mnist" ) {
            config.dataDir = "../data/mnist";
            config.trainFile = "train-dat.mat";
            config.validateFile = "t10k-dat.mat";
        } else if( dataset == "norb" ) {
            config.dataDir = "../data/norb";
            config.trainFile = "training-shuffled-dat.mat";
            config.validateFile = "testing-sampled-dat.mat";
        } else if( dataset == "cifar10" ) {
            config.dataDir = "../data/cifar10";
            config.trainFile = "train-dat.mat";
            config.validateFile = "test-dat.mat";
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


