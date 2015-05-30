// Copyright Hugh Perkins (hughperkins at gmail), Josef Moudrik 2015
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "DeepCL.h"
#include "loss/SoftMaxLayer.h"

using namespace std;

/* [[[cog
    # These are used in the later cog sections in this file:
    # format:
    # ( name, type, description, default, ispublicapi )
    options = [
        ('gpuIndex', 'int', 'gpu device index; default value is gpu if present, cpu otw.', -1, True),

        ('weightsFile', 'string', 'file to read weights from','weights.dat', True),
        # removing loadondemand for now, let's always load exactly one batch at a time for now
        # ('loadOnDemand', 'int', 'load data on demand [1|0]', 0, True),
        ('batchSize', 'int', 'batch size', 128, True),

        # lets go with pipe for now, and then somehow shoehorn files in later?
        ('inputFile',  'string', 'file to read inputs from, if empty, read stdin (default)', '', False),
        ('outputFile', 'string', 'file to write outputs to, if empty, write to stdout', '', False),
        ('writeLabels', 'int', 'write integer labels, instead of probabilities etc (default 0)', 0, False),
        ('outputFormat', 'string', 'output format [binary|text]', 'text', False)
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
    int batchSize;
    string inputFile;
    string outputFile;
    int writeLabels;
    string outputFormat;
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
        batchSize = 128;
        inputFile = "";
        outputFile = "";
        writeLabels = 0;
        outputFormat = "text";
        // [[[end]]]
    }
};

void go(Config config) {
    int N = -1;
    int numPlanes;
    int imageSize;
    int imageSizeCheck;
    if( config.inputFile == "" ) {
        int dims[3];
        cin.read( reinterpret_cast< char * >( dims ), 3 * 4l );
        numPlanes = dims[0];
        imageSize = dims[1];
        imageSizeCheck = dims[2];
        if( imageSize != imageSizeCheck ) {
            throw std::runtime_error( "imageSize doesnt match imageSizeCheck, image not square" );
        }
    } else {
        GenericLoader::getDimensions( config.inputFile, &N, &numPlanes, &imageSize );
    }
//    cout << "planes " << numPlanes << " size " << imageSize << " sizecheck " << imageSizeCheck << endl;

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
    if ( !WeightsPersister::loadConfigString( config.weightsFile, netDef ) ){
        cout << "Cannot load network definition from weightsFile." << endl;
        return;
    }
//    cout << "net def from weights file: " << netDef << endl;

    net->addLayer( InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize) );
    net->addLayer( NormalizationLayerMaker::instance()->translate( 0.0f )->scale( 1.0f ) ); // This will be read from weights file

    if( !NetdefToNet::createNetFromNetdef( net, netDef, weightsInitializer ) ) {
        return;
    }

    // ignored int and float, s.t. we can use loadWeights
    int ignI;
    float ignF;

    // weights file contains normalization layer parameters as 'weights' now.  We should probably rename weights to parameters
    // sooner or later ,but anyway, tehcnically, works for onw
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

    float *inputData = new float[ inputCubeSize * config.batchSize];

    int *labels = new int[config.batchSize];
    int n = 0;
    bool more = true;
    if( config.inputFile == "" ) {
        #ifdef _WIN32
        // refs:
        // http://www.thecodingforums.com/threads/binary-output-to-stdout-in-windows.317367/
        // http://www.cplusplus.com/forum/windows/77812/
        _setmode( _fileno( stdout ), _O_BINARY ); 
        #endif
        cin.read( reinterpret_cast< char * >( inputData ), inputCubeSize * config.batchSize * 4l );
        more = cin;
    } else {
        // pass 0 for labels, and this will cause GenericLoader to simply not try to load any labels
        // now, after modifying GenericLoader to have this new behavior
        GenericLoader::load( config.inputFile, inputData, 0, n, config.batchSize );
    }
    ostream *outFile = 0;
    if( config.outputFile == "" ) {
        outFile = &cout;
    } else {
        if( config.outputFormat == "text" ) {
            outFile = new ofstream( config.outputFile, ios::out );
        } else {
            outFile = new ofstream( config.outputFile, ios::out | std::ios::binary );
        }
    }
    while( more ) {
        net->forward(inputData);

        if( !config.writeLabels ) {
            if( config.outputFormat == "text" ) {
                float const*output = net->getOutput();
                const int numFields = net->getLastLayer()->getOutputCubeSize();
                for( int i = 0; i < config.batchSize; i++ ) {
                    for( int f = 0; f < numFields; f++ ) {
                        if( f > 0 ) {
                            *outFile << " ";
                        }
                        *outFile << output[ i * numFields + f ];
                    }
                }
                *outFile << "\n";
            } else {
                outFile->write( reinterpret_cast<const char *>(net->getOutput()), net->getOutputSize() * 4 * config.batchSize);
            }
        } else {
            SoftMaxLayer *softMaxLayer = dynamic_cast< SoftMaxLayer *>(net->getLastLayer() );
            if( softMaxLayer == 0 ) {
                cout << "must have softmaxlayer as last layer, if want to output labels" << endl;
                return;
            }
            softMaxLayer->getLabels(labels);
            if( config.outputFormat == "text" ) {
                for( int i = 0; i < config.batchSize; i++ ) {
                    *outFile << labels[i] << "\n";
                }
            } else {
                outFile->write( reinterpret_cast< char * >( labels ), config.batchSize * 4l );
            }
            outFile->flush();
        }
        n += config.batchSize;
        n += config.batchSize;
        if( config.inputFile == "" ) {
            cin.read( reinterpret_cast< char * >( inputData ), inputCubeSize * config.batchSize * 4l );
            more = cin;
        } else {
            if( n + config.batchSize < N ) {
                GenericLoader::load( config.inputFile, inputData, 0, n, config.batchSize );
            } else {
                more = false;
                if( n != N ) {
                    cout << "breaking prematurely, since file is not an exact multiple of batchsize, and we didnt handle this yet" << endl;
                }
            }
        }
    }
    if( config.outputFile != "" ) {
        delete outFile;
    }

    delete[] inputData;
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
    cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
    cout << "" << endl; 
    cout << "unstable, might change within major version:" << endl; 
    cout << "    inputfile=[file to read inputs from, if empty, read stdin (default)] (" << config.inputFile << ")" << endl;
    cout << "    outputfile=[file to write outputs to, if empty, write to stdout] (" << config.outputFile << ")" << endl;
    cout << "    writelabels=[write integer labels, instead of probabilities etc (default 0)] (" << config.writeLabels << ")" << endl;
    cout << "    outputformat=[output format [binary|text]] (" << config.outputFormat << ")" << endl;
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
            } else if( key == "batchsize" ) {
                config.batchSize = atoi(value);
            } else if( key == "inputfile" ) {
                config.inputFile = (value);
            } else if( key == "outputfile" ) {
                config.outputFile = (value);
            } else if( key == "writelabels" ) {
                config.writeLabels = atoi(value);
            } else if( key == "outputformat" ) {
                config.outputFormat = (value);
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


