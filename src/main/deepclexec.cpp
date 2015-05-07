// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


//#include <iostream>
//#include <algorithm>

#include "DeepCL.h"

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

        ('inputFile',  'string', 'file to read inputs from', '', True),
        ('outputFile', 'string', 'file to write outputs to', '', True)
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
    int batchSize;
    string dataDir;
    string trainFile;
    string dataset;
    string weightsFile;
    string normalization;
    float normalizationNumStds;
    int loadOnDemand;
    int normalizationExamples;
    string inputFile;
    string outputFile;
    // [[[end]]]

    string netDef;

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
        loadOnDemand = 0;
        normalizationExamples = 10000;
        batchSize = 128;
        inputFile = "";
        outputFile = "";
        // [[[end]]]

        netDef = "";
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

    // wont be necessary anymore
    if( trainData != 0 ) {
        delete[] trainData;
        trainData = 0;
    }
    if( trainLabels != 0 ) {
        delete[] trainLabels;
        trainLabels = 0;
    }
    
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

    if( !WeightsPersister::loadWeights( config.weightsFile, netDef, net, &ignI, &ignI, &ignF, &ignI, &ignF ) ){
        cout << "Cannot load network weights from weightsFile." << endl;
        return;
    }

    net->print();
    net->setBatchSize(1);

    delete weightsInitializer;

    //
    // ## All is set up now
    // 


    float * inputData = new float[ (long) inputCubeSize];
    ifstream fin(config.inputFile, ios::in | ios::binary);
    ofstream fout(config.outputFile, ios::out | ios::binary);

    if( ! fin ){
        cout << "Cannot open input file." << endl;
        return;
    }

    if( ! fout ){
        cout << "Cannot open output file." << endl;
        return;
    }


    cout << "Reading inputs from:  '" << config.inputFile << "'" << endl;
    cout << "Writing outputs from: '" << config.outputFile << "'" << endl;

    while( true ){
        fin.read( reinterpret_cast<char *>(inputData), inputCubeSize);
        if( !fin );
            break;

        cout << "Read one input cube." << endl;

        fout.write("Bleble\n", 7);
        if( !fout );
            break;

            /*
        net->forward(inputData);

        fout.write( reinterpret_cast<const char *>(net->getOutput()), net->getOutputSize());
        if( !fout );
            break;
            */
    }

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
    cout << "    inputFile=[path to file with input data (e.g. a named pipe)] (" << config.inputFile << ")" << endl;
    cout << "    outputFile=[path to file to write output data to (e.g. a named pipe)] (" << config.outputFile << ")" << endl;
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
            } else if( key == "loadondemand" ) {
                config.loadOnDemand = atoi(value);
            } else if( key == "normalizationexamples" ) {
                config.normalizationExamples = atoi(value);
            } else if( key == "inputfile" ) {
                config.inputFile = (value);
            } else if( key == "outputfile" ) {
                config.outputFile = (value);
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


