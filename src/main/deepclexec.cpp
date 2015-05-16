// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


//#include <iostream>
//#include <algorithm>

#include "DeepCL.h"
#include "loss/SoftMaxLayer.h"
//#include "test/Sampler.h"

using namespace std;

/* [[[cog
    # These are used in the later cog sections in this file:
    # format:
    # ( name, type, description, default, ispublicapi )
    options = [
        ('gpuIndex', 'int', 'gpu device index; default value is gpu if present, cpu otw.', -1, True),

        ('dataDir', 'string', 'directory to search for train and validate files', '../data/mnist', True),
        ('trainFile', 'string', 'path to training data file',"train-images-idx3-ubyte", True),
        ('dataset', 'string', 'choose datadir,trainfile,and validatefile for certain datasets [mnist|norb|kgsgo|cifar10]','', True),
        ('weightsFile', 'string', 'file to read weights from','weights.dat', True),
        ('normalization', 'string', '[stddev|maxmin]', 'stddev', True),
        ('normalizationNumStds', 'float', 'with stddev normalization, how many stddevs from mean is 1?', 2.0, True),
        ('normalizationExamples', 'int', 'number of examples to read to determine normalization parameters', 10000, True),
        ('loadOnDemand', 'int', 'load data on demand [1|0]', 0, True),
        ('batchSize', 'int', 'batch size', 128, True),

        ('inputFile',  'string', 'file to read inputs from', 'input.dat', True),
        ('outputFile', 'string', 'file to write outputs to', 'output.dat', True),
        ('writeIntLabels', 'int', 'write integer labels, instead of probabilities etc (default 0)', 0, False)
    ]
*///]]]
// [[[end]]]

class Config {
public:
    /* [[[cog
        cog.outl('// generated using cog:')
        for (name,type,description,default,_) in options:
            cog.outl( type + ' ' + name + ';')
    */// ]]]
    // generated using cog:
    int gpuIndex;
    string dataDir;
    string trainFile;
    string dataset;
    string weightsFile;
    string normalization;
    float normalizationNumStds;
    int normalizationExamples;
    int loadOnDemand;
    int batchSize;
    string inputFile;
    string outputFile;
    int writeIntLabels;
    // [[[end]]]

    Config() {
        /* [[[cog
            cog.outl('// generated using cog:')
            for (name,type,description,default,_) in options:
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
        gpuIndex = -1;
        dataDir = "../data/mnist";
        trainFile = "train-images-idx3-ubyte";
        dataset = "";
        weightsFile = "weights.dat";
        normalization = "stddev";
        normalizationNumStds = 2.0f;
        normalizationExamples = 10000;
        loadOnDemand = 0;
        batchSize = 128;
        inputFile = "input.dat";
        outputFile = "output.dat";
        writeIntLabels = 0;
        // [[[end]]]
    }
};

void go(Config config) {
    int Ntrain;
    int numPlanes;
    int imageSize;

    float *trainData = 0;
    int *trainLabels = 0;
    int trainAllocateN = 0;

    //
    // ## Load training data (for initialization of normalizer)
    //

    GenericLoader::getDimensions( config.dataDir + "/" + config.trainFile, &Ntrain, &numPlanes, &imageSize );

    // we need just the number of examples to compute the normalization params
    Ntrain = config.normalizationExamples > Ntrain ? Ntrain : config.normalizationExamples;

    cout << "normalizationExamples " << Ntrain << " numPlanes " << numPlanes << " imageSize " << imageSize << endl;
    if( config.loadOnDemand ) {
        trainAllocateN = config.batchSize; // can improve this later
    } else {
        trainAllocateN = Ntrain;
    }

    const long inputCubeSize = numPlanes * imageSize * imageSize ;

    trainData = new float[ (long)trainAllocateN * inputCubeSize];
    trainLabels = new int[trainAllocateN];
    if( !config.loadOnDemand && Ntrain > 0 ) {
        GenericLoader::load( config.dataDir + "/" + config.trainFile, trainData, trainLabels, 0, Ntrain );
    }

    //
    // ## Init the normalizer
    //

    float translate;
    float scale;
    if( !config.loadOnDemand ) {
        if( config.normalization == "stddev" ) {
            float mean, stdDev;
            NormalizationHelper::getMeanAndStdDev( trainData, Ntrain * inputCubeSize, &mean, &stdDev );
            cout << " image stats mean " << mean << " stdDev " << stdDev << endl;
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
            NormalizeGetStdDev normalizeGetStdDev( trainData, trainLabels ); 
            BatchProcess::run( config.dataDir + "/" + config.trainFile, 0, config.batchSize, Ntrain, inputCubeSize, &normalizeGetStdDev );
            normalizeGetStdDev.calcMeanStdDev( &mean, &stdDev );
            cout << " image stats mean " << mean << " stdDev " << stdDev << endl;
            translate = - mean;
            scale = 1.0f / stdDev / config.normalizationNumStds;
        } else if( config.normalization == "maxmin" ) {
            NormalizeGetMinMax normalizeGetMinMax( trainData, trainLabels );
            BatchProcess::run( config.dataDir + "/" + config.trainFile, 0, config.batchSize, Ntrain, inputCubeSize, &normalizeGetMinMax );
            normalizeGetMinMax.calcMinMaxTransform( &translate, &scale );
        } else {
            cout << "Error: Unknown normalization: " << config.normalization << endl;
            return;
        }
    }
    cout << " image norm translate " << translate << " scale " << scale << endl;

    //
    // ## Set up the Network
    //

    EasyCL *cl = 0;
    if( config.gpuIndex >= 0 ) {
        cl = EasyCL::createForIndexedGpu( config.gpuIndex );
    } else {
        cl = EasyCL::createForFirstGpuOtherwiseCpu();
    }

    NeuralNet *net;
    net = new NeuralNet(cl);

    // just use the default for net creation, weights are overriden from the weightsFile
    // yes, thats fine - Hugh
    WeightsInitializer *weightsInitializer = new OriginalInitializer();

    if( config.weightsFile == "" ) {
        cout << "weightsFile not specified" << endl;
        return;
    }

    string netDef;
    if ( ! WeightsPersister::loadConfigString(config.weightsFile, netDef)){
        cout << "Cannot load network definition from weightsFile." << endl;
        return;
    }

    net->addLayer( InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize) );
    net->addLayer( NormalizationLayerMaker::instance()->translate(translate)->scale(scale) );

    if( !NetdefToNet::createNetFromNetdef( net, netDef, weightsInitializer ) ) {
        return;
    }

    // ignored int and float, s.t. we can use loadWeights
    int ignI;
    float ignF;

    if( !WeightsPersister::loadWeights( config.weightsFile, string("netDef=")+netDef, net, &ignI, &ignI, &ignF, &ignI, &ignF ) ){
        cout << "Cannot load network weights from weightsFile." << endl;
        return;
    }

//    float *weights = net->getWeights();
//    float *bias = net->getBias();
//    Sampler::sampleFloatWrapper( "weights", net->getLayer(6)->getWeightsWrapper() );
//    Sampler::sampleFloatWrapper( "weights", net->getLayer(11)->getWeightsWrapper() );

    net->print();  // this output should match what you trained on  - Hugh
    net->setBatchSize(config.batchSize);  // 1? that cant be right?  changed to read from config  - Hugh
    cout << "batchSize: " << config.batchSize << endl;


    //
    // ## All is set up now
    // 


    // ideally, this could go in GenericReader somehow I reckon, but putting it here is ok for now :-)   - Hugh
    // I'm going to test with mnist, since it's small and gives nice results, and fast :-)   - Hugh
    float * inputData = new float[ inputCubeSize * config.batchSize];
    if( config.dataDir != "" ) {
        config.inputFile = config.dataDir + "/" + config.inputFile;
        config.outputFile = config.dataDir + "/" + config.outputFile;
    }
    cout << "Reading inputs from:  '" << config.inputFile << "'" << endl;
    cout << "Writing outputs to: '" << config.outputFile << "'" << endl;
//    ifstream fin(config.inputFile, ios::in | ios::binary);
//    ofstream fout(config.outputFile, ios::out | ios::binary);
    // sorry I dont know how to use ifstream, so I'm going to use filehelper, cos I know it works :-)   - Hugh
   // ( but re-wrtiing it back to use fstream is probably a good idea )

//    if( ! fin ){
//        cout << "Cannot open input file: '"<< config.inputFile <<"'" << endl;
//        return;
//    }

//    if( ! fout ){
//        cout << "Cannot open output file: '"<< config.outputFile <<"'" << endl;
//        return;
//    }


    cout << "Input cube size is: " << inputCubeSize * 4 << " B" << endl;
    cout << "Output image size is: " << net->getOutputCubeSize() * 4 << " B" << endl;

    int *labels = new int[config.batchSize];
    int n = 0;
    long fileSize = FileHelper::getFilesize( config.inputFile );
    int totalN = fileSize / inputCubeSize / 4;
    cout << "totalN: " << totalN << endl;
    while( n < totalN ){
//        fin.read( reinterpret_cast<char *>(inputData), config.batchSize * inputCubeSize * 4);
        FileHelper::readBinaryChunk( reinterpret_cast<char *>(inputData), config.inputFile, (long)n * inputCubeSize * 4, (long)inputCubeSize * config.batchSize * 4 );
//        if( !fin ){
//            break;
//        }

    	cout << "Read " << config.batchSize << " input cubes." << endl;

        net->forward(inputData);  // seems ok...   - Hugh

        if( !config.writeIntLabels ) {
            FileHelper::writeBinaryChunk( config.outputFile, reinterpret_cast<const char *>(net->getOutput()), 
                n * 4 * net->getOutputCubeSize(),
                config.batchSize * 4 * net->getOutputCubeSize() );
//            fout.write( reinterpret_cast<const char *>(net->getOutput()), net->getOutputSize() * 4 * config.batchSize);
        } else {
            // calculate the labels somehow...
            // ... added 'getLabels' to SoftMaxLayer
            SoftMaxLayer *softMaxLayer = dynamic_cast< SoftMaxLayer *>(net->getLastLayer() );
            if( softMaxLayer == 0 ) {
                cout << "must have softmaxlayer as last layer, if want to output labels" << endl;
                return;
            }
            softMaxLayer->getLabels(labels);
//            fout.write( reinterpret_cast<const char *>(labels), config.batchSize * 4);
            if( n == 0 ) {
                for( int i = 0; i < config.batchSize / 4; i++ ) {
                    cout << "out[" << i << "]=" << labels[i] << endl;
                }
            }
            FileHelper::writeBinaryChunk( config.outputFile, reinterpret_cast<const char *>(labels), n * 4, config.batchSize * 4);
        }
        n += config.batchSize;
        if( ( n + config.batchSize > totalN ) && ( n != totalN ) ) {
            cout << "breaking prematurely, since file is not an exact multiple of batchsize, and we didnt handle this yet" << endl;
            break;
        }
//        if( !fout ){
//            break;
//        }

//	fout.flush();
//        cout << "Written output image." << endl;
    }

    cout << "Exiting." << endl;

    delete[] labels;
    delete weightsInitializer; // I'm not entirely trusting of my delete sections, so let's put 
                               // deletes at the end, here  :-P - Hugh
    delete net;
    delete cl;
}

void printUsage( char *argv[], Config config ) {
    cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
    cout << endl;
    cout << "Possible key=value pairs:" << endl;
    /* [[[cog
        cog.outl('// generated using cog:')
        cog.outl('cout << "public api, shouldnt change within major version:" << endl;')
        for (name,type,description,_, is_public_api) in options:
            if is_public_api:
                cog.outl( 'cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
        cog.outl('cout << "" << endl; ')
        cog.outl('cout << "unstable, might change within major version:" << endl; ')
        for (name,type,description,_, is_public_api) in options:
            if not is_public_api:
                cog.outl( 'cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
    *///]]]
    // generated using cog:
    cout << "public api, shouldnt change within major version:" << endl;
    cout << "    gpuindex=[gpu device index; default value is gpu if present, cpu otw.] (" << config.gpuIndex << ")" << endl;
    cout << "    datadir=[directory to search for train and validate files] (" << config.dataDir << ")" << endl;
    cout << "    trainfile=[path to training data file] (" << config.trainFile << ")" << endl;
    cout << "    dataset=[choose datadir,trainfile,and validatefile for certain datasets [mnist|norb|kgsgo|cifar10]] (" << config.dataset << ")" << endl;
    cout << "    weightsfile=[file to read weights from] (" << config.weightsFile << ")" << endl;
    cout << "    normalization=[[stddev|maxmin]] (" << config.normalization << ")" << endl;
    cout << "    normalizationnumstds=[with stddev normalization, how many stddevs from mean is 1?] (" << config.normalizationNumStds << ")" << endl;
    cout << "    normalizationexamples=[number of examples to read to determine normalization parameters] (" << config.normalizationExamples << ")" << endl;
    cout << "    loadondemand=[load data on demand [1|0]] (" << config.loadOnDemand << ")" << endl;
    cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
    cout << "    inputfile=[file to read inputs from] (" << config.inputFile << ")" << endl;
    cout << "    outputfile=[file to write outputs to] (" << config.outputFile << ")" << endl;
    cout << "" << endl; 
    cout << "unstable, might change within major version:" << endl; 
    cout << "    writeintlabels=[write integer labels, instead of probabilities etc (default 0)] (" << config.writeIntLabels << ")" << endl;
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
                for (name,type,description,_,_) in options:
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
            } else if( key == "gpuindex" ) {
                config.gpuIndex = atoi(value);
            } else if( key == "datadir" ) {
                config.dataDir = (value);
            } else if( key == "trainfile" ) {
                config.trainFile = (value);
            } else if( key == "dataset" ) {
                config.dataset = (value);
            } else if( key == "weightsfile" ) {
                config.weightsFile = (value);
            } else if( key == "normalization" ) {
                config.normalization = (value);
            } else if( key == "normalizationnumstds" ) {
                config.normalizationNumStds = atof(value);
            } else if( key == "normalizationexamples" ) {
                config.normalizationExamples = atoi(value);
            } else if( key == "loadondemand" ) {
                config.loadOnDemand = atoi(value);
            } else if( key == "batchsize" ) {
                config.batchSize = atoi(value);
            } else if( key == "inputfile" ) {
                config.inputFile = (value);
            } else if( key == "outputfile" ) {
                config.outputFile = (value);
            } else if( key == "writeintlabels" ) {
                config.writeIntLabels = atoi(value);
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
        } else if( dataset == "norb" ) {
            config.dataDir = "../data/norb";
            config.trainFile = "training-shuffled-dat.mat";
        } else if( dataset == "cifar10" ) {
            config.dataDir = "../data/cifar10";
            config.trainFile = "train-dat.mat";
        } else if( dataset == "kgsgo" ) {
            config.dataDir = "../data/kgsgo";
            config.trainFile = "kgsgo-train10k-v2.dat";
            config.loadOnDemand = 1;
        } else if( dataset == "kgsgoall" ) {
            config.dataDir = "../data/kgsgo";
            config.trainFile = "kgsgo-trainall-v2.dat";
            config.loadOnDemand = 1;
        } else {
            cout << "dataset " << dataset << " not known.  please choose from: mnist, norb, cifar10, kgsgo" << endl;
            return -1;
        }
        cout << "Using dataset " << dataset << ":" << endl;
        cout << "   datadir: " << config.dataDir << ":" << endl;
        cout << "   trainfile: " << config.trainFile << ":" << endl;
    }
    try {
        go( config );
    } catch( runtime_error e ) {
        cout << "Something went wrong: " << e.what() << endl;
        return -1;
    }
}


