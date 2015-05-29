// Copyright Hugh Perkins (hughperkins at gmail), Josef Moudrik 2015
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

        ('weightsFile', 'string', 'file to read weights from','weights.dat', True),
        ('loadOnDemand', 'int', 'load data on demand [1|0]', 0, True),
        ('batchSize', 'int', 'batch size', 128, True),

        # lets go with pipe for now, and then somehow shoehorn files in later?
        # ('inputFile',  'string', 'file to read inputs from', 'input.dat', False),
        # ('outputFile', 'string', 'file to write outputs to', 'output.dat', True),
        ('writeIntLabels', 'int', 'write integer labels, instead of probabilities etc (default 0)', 0, True)
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
    string weightsFile;
    int loadOnDemand;
    int batchSize;
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
        weightsFile = "weights.dat";
        loadOnDemand = 0;
        batchSize = 128;
        writeIntLabels = 0;
        // [[[end]]]
    }
};

void go(Config config) {
//    int N;
    int numPlanes;
    int imageSize;
    int imageSizeCheck;
    int dims[3];
    cin.read( reinterpret_cast< char * >( dims ), 3 * 4l );
    numPlanes = dims[0];
    imageSize = dims[1];
    imageSizeCheck = dims[2];
//    cout << "planes " << numPlanes << " size " << imageSize << " sizecheck " << imageSizeCheck << endl;
    if( imageSize != imageSizeCheck ) {
        throw std::runtime_error( "imageSize doesnt match imageSizeCheck, image not square" );
    }

//    float *data = 0;
//    int allocateN = 0;

    //
    // ## Load training data (for initialization of normalizer)
    //

    // we need just the number of examples to compute the normalization params
//    N = config.normalizationExamples > N ? N : config.normalizationExamples;

    const long inputCubeSize = numPlanes * imageSize * imageSize ;

    //
    // ## Set up the Network
    //

    EasyCL *cl = 0;
    if( config.gpuIndex >= 0 ) {
        cl = EasyCL::createForIndexedGpu( config.gpuIndex, false );
    } else {
        cl = EasyCL::createForFirstGpuOtherwiseCpu(false);
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
    float translate;
    float scale;
    if ( !WeightsPersister::loadConfigString( config.weightsFile, netDef, &translate, &scale ) ){
        cout << "Cannot load network definition from weightsFile." << endl;
        return;
    }
//    cout << "net def from weights file: " << netDef << endl;

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

    //net->print();
    net->setBatchSize(config.batchSize);
//    cout << "batchSize: " << config.batchSize << endl;


    //
    // ## All is set up now
    // 


    // ideally, this could go in GenericReader somehow I reckon, but putting it here is ok for now :-)   - Hugh
    float * inputData = new float[ inputCubeSize * config.batchSize];
// jm: this makes it impossible to select iofiles outside the datadir,
//     which we should be able to do
//
//
//    if( config.dataDir != "" ) {
//        config.inputFile = config.dataDir + "/" + config.inputFile;
//        config.outputFile = config.dataDir + "/" + config.outputFile;
//    }
//    cout << "Reading inputs from:  '" << config.inputFile << "'" << endl;
//    cout << "Writing outputs to: '" << config.outputFile << "'" << endl;
//    ifstream fin(config.inputFile, ios::in | ios::binary);
//    ofstream fout(config.outputFile, ios::out | ios::binary);

//    if( ! fin ){
//        cout << "Cannot open input file: '"<< config.inputFile <<"'" << endl;
//        return;
//    }

//    if( ! fout ){
//        cout << "Cannot open output file: '"<< config.outputFile <<"'" << endl;
//        return;
//    }


//    cout << "Input cube size is: " << inputCubeSize * 4 << " B" << endl;
//    cout << "Output image size is: " << net->getOutputCubeSize() * 4 << " B" << endl;

    int *labels = new int[config.batchSize];
    int n = 0;
//    long fileSize = FileHelper::getFilesize( config.inputFile );
//    int totalN = fileSize / inputCubeSize / 4;
//    cout << "totalN: " << totalN << endl;
//    while( n < totalN ){
    
    #ifdef _WIN32
    // refs:
    // http://www.thecodingforums.com/threads/binary-output-to-stdout-in-windows.317367/
    // http://www.cplusplus.com/forum/windows/77812/
    _setmode( _fileno( stdout ), _O_BINARY ); 
    #endif
    cin.read( reinterpret_cast< char * >( inputData ), inputCubeSize * config.batchSize * 4l );
    while( cin ) {
//        fin.read( reinterpret_cast<char *>(inputData), config.batchSize * inputCubeSize * 4);
//        FileHelper::readBinaryChunk( reinterpret_cast<char *>(inputData), config.inputFile, (long)n * inputCubeSize * 4, (long)inputCubeSize * config.batchSize * 4 );
//        if( !fin ){
//            break;
//        }

//    	cout << "Read " << config.batchSize << " input cubes." << endl;

        net->forward(inputData);  // seems ok...   - Hugh

        if( !config.writeIntLabels ) {
//            FileHelper::writeBinaryChunk( config.outputFile, reinterpret_cast<const char *>(net->getOutput()), 
//                n * 4 * net->getOutputCubeSize(),
//                config.batchSize * 4 * net->getOutputCubeSize() );
//            fout.write( reinterpret_cast<const char *>(net->getOutput()), net->getOutputSize() * 4 * config.batchSize);
            cout.write( reinterpret_cast<const char *>(net->getOutput()), net->getOutputSize() * 4 * config.batchSize);
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
//            if( n == 0 ) {
//                for( int i = 0; i < config.batchSize / 4; i++ ) {
//                    cout << "out[" << i << "]=" << labels[i] << endl;
//                }
//            }
            cout.write( reinterpret_cast< char * >( labels ), config.batchSize * 4l );
//            FileHelper::writeBinaryChunk( config.outputFile, reinterpret_cast<const char *>(labels), n * 4, config.batchSize * 4);
        }
        n += config.batchSize;
//        if( ( n + config.batchSize > totalN ) && ( n != totalN ) ) {
//            cout << "breaking prematurely, since file is not an exact multiple of batchsize, and we didnt handle this yet" << endl;
//            break;
//        }
//        if( !fout ){
//            break;
//        }

//	fout.flush();
//        cout << "Written output image." << endl;
        cin.read( reinterpret_cast< char * >( inputData ), inputCubeSize * config.batchSize * 4l );
    }

//    cout << "Exiting." << endl;

    delete[] labels;
    delete weightsInitializer;
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
    cout << "    weightsfile=[file to read weights from] (" << config.weightsFile << ")" << endl;
    cout << "    loadondemand=[load data on demand [1|0]] (" << config.loadOnDemand << ")" << endl;
    cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
    cout << "    writeintlabels=[write integer labels, instead of probabilities etc (default 0)] (" << config.writeIntLabels << ")" << endl;
    cout << "" << endl; 
    cout << "unstable, might change within major version:" << endl; 
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
            } else if( key == "weightsfile" ) {
                config.weightsFile = (value);
            } else if( key == "loadondemand" ) {
                config.loadOnDemand = atoi(value);
            } else if( key == "batchsize" ) {
                config.batchSize = atoi(value);
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
    try {
        go( config );
    } catch( runtime_error e ) {
        cout << "Something went wrong: " << e.what() << endl;
        return -1;
    }
}


