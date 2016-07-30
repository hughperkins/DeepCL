// Copyright Hugh Perkins (hughperkins at gmail), Josef Moudrik 2015
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.


#include "DeepCL.h"
#include "loss/SoftMaxLayer.h"
#ifdef _WIN32
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#endif // _WIN32
#include "clblas/ClBlasInstance.h"

using namespace std;

/* [[[cog
    # These are used in the later cog sections in this file:
    options = [
        {'name': 'gpuIndex', 'type': 'int', 'description': 'gpu device index; default value is gpu if present, cpu otw.', 'default': -1, 'ispublicapi': True},

        {'name': 'weightsFile', 'type': 'string', 'description': 'file to read weights from', 'default': 'weights.dat', 'ispublicapi': True},
        # removing loadondemand for now, let's always load exactly one batch at a time for now
        # ('loadOnDemand', 'int', 'load data on demand [1|0]', 0, [0,1], True},
        {'name': 'batchSize', 'type': 'int', 'description': 'batch size', 'default': 128, 'ispublicapi': True},

        # lets go with pipe for now, and then somehow shoehorn files in later?
        {'name': 'inputFile',  'type': 'string', 'description': 'file to read inputs from, if empty, read stdin (default)', 'default': ''},
        {'name': 'outputFile', 'type': 'string', 'description': 'file to write outputs to, if empty, write to stdout', 'default': ''},
        {'name': 'outputLayer', 'type': 'int', 'description': 'layer to write output from, default -1 means: last layer', 'default': -1},
        {'name': 'writeLabels', 'type': 'int', 'description': 'write integer labels, instead of probabilities etc (default 0)', 'default': 0},
        {'name': 'outputFormat', 'type': 'string', 'description': 'output format [binary|text]', 'default': 'text'}
    ]
*///]]]
// [[[end]]]

class Config {
public:
    /* [[[cog
        cog.outl('// generated using cog:')
        for option in options:
            cog.outl(option['type'] + ' ' + option['name'] + ';')
    */// ]]]
    // generated using cog:
    int gpuIndex;
    string weightsFile;
    int batchSize;
    string inputFile;
    string outputFile;
    int outputLayer;
    int writeLabels;
    string outputFormat;
    // [[[end]]]

    Config() {
        /* [[[cog
            cog.outl('// generated using cog:')
            for option in options:
                defaultString = ''
                default = option['default']
                type = option['type']
                if type == 'string':
                    defaultString = '"' + default + '"'
                elif type == 'int':
                    defaultString = str(default)
                elif type == 'float':
                    defaultString = str(default)
                    if '.' not in defaultString:
                        defaultString += '.0'
                    defaultString += 'f'
                cog.outl(option['name'] + ' = ' + defaultString + ';')
        */// ]]]
        // generated using cog:
        gpuIndex = -1;
        weightsFile = "weights.dat";
        batchSize = 128;
        inputFile = "";
        outputFile = "";
        outputLayer = -1;
        writeLabels = 0;
        outputFormat = "text";
        // [[[end]]]
    }
};

void go(Config config) {
    bool verbose = true;
    if(config.outputFile == "") {
        verbose = false;
    }

    int N = -1;
    int numPlanes;
    int imageSize;
    int imageSizeCheck;
    GenericLoaderv2* loader = NULL;
    if(config.inputFile == "") {
        int dims[3];
        cin.read(reinterpret_cast< char * >(dims), 3 * 4l);
        numPlanes = dims[0];
        imageSize = dims[1];
        imageSizeCheck = dims[2];
        if(imageSize != imageSizeCheck) {
            throw std::runtime_error("imageSize doesnt match imageSizeCheck, image not square");
        }
    } else {
        loader = new GenericLoaderv2(config.inputFile);
        N = loader->getN();
        numPlanes = loader->getPlanes();
        imageSize = loader->getImageSize();
        // GenericLoader::getDimensions(config.inputFile.c_str(), &N, &numPlanes, &imageSize);
        if(verbose) cout << "N " << N << " planes " << numPlanes << " size " << imageSize << endl;
    }

    const long inputCubeSize = numPlanes * imageSize * imageSize ;

    //
    // ## Set up the Network
    //

    EasyCL *cl = 0;
    if(config.gpuIndex >= 0) {
        cl = EasyCL::createForIndexedGpu(config.gpuIndex, verbose);
    } else {
        cl = EasyCL::createForFirstGpuOtherwiseCpu(verbose);
    }
    ClBlasInstance blasInstance;

    NeuralNet *net;
    net = new NeuralNet(cl);

    // just use the default for net creation, weights are overriden from the weightsFile
    WeightsInitializer *weightsInitializer = new OriginalInitializer();

    if(config.weightsFile == "") {
        cout << "weightsFile not specified" << endl;
        return;
    }

    string netDef;
    if (!WeightsPersister::loadConfigString(config.weightsFile, netDef) ){
        cout << "Cannot load network definition from weightsFile." << endl;
        return;
    }
//    cout << "net def from weights file: " << netDef << endl;

    net->addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize));
    net->addLayer(NormalizationLayerMaker::instance()->translate(0.0f)->scale(1.0f) ); // This will be read from weights file

    if(!NetdefToNet::createNetFromNetdef(net, netDef, weightsInitializer) ) {
        return;
    }

    // ignored int and float, s.t. we can use loadWeights
    int ignI;
    float ignF;

    // weights file contains normalization layer parameters as 'weights' now.  We should probably rename weights to parameters
    // sooner or later ,but anyway, tehcnically, works for onw
    if(!WeightsPersister::loadWeights(config.weightsFile, string("netDef=")+netDef, net, &ignI, &ignI, &ignF, &ignI, &ignF) ){
        cout << "Cannot load network weights from weightsFile." << endl;
        return;
    }

    if(verbose) {
        net->print();
    }
    net->setBatchSize(config.batchSize);
    if(verbose) cout << "batchSize: " << config.batchSize << endl;


    //
    // ## All is set up now
    //

    float *inputData = new float[ inputCubeSize * config.batchSize];

    int *labels = new int[config.batchSize];
    int n = 0;
    bool more = true;
    ostream *outFile = 0;
    if(verbose) cout << "outputFile: '" << config.outputFile << "'"<< endl;
    if(config.outputFile == "") {
        #ifdef _WIN32
        // refs:
        // http://www.thecodingforums.com/threads/binary-output-to-stdout-in-windows.317367/
        // http://www.cplusplus.com/forum/windows/77812/
        _setmode(_fileno(stdout), _O_BINARY);
        #endif
        outFile = &cout;
    } else {
        if(config.outputFormat == "text") {
            outFile = new ofstream(config.outputFile, ios::out);
        } else if(config.outputFormat == "binary") {
            outFile = new ofstream(config.outputFile, ios::out | std::ios::binary);
        } else {
            throw runtime_error("outputFormat " + config.outputFormat + " not recognized");
        }
    }
    if(config.outputLayer == -1) {
        config.outputLayer = net->getNumLayers() - 1;
    }
    if(verbose) cout << "inputFile: '" << config.inputFile << "'"<< endl;
    if(config.inputFile == "") {
        cin.read(reinterpret_cast< char * >(inputData), inputCubeSize * config.batchSize * 4l);
        more = !cin.eof();
    } else {
        // pass 0 for labels, and this will cause GenericLoader to simply not try to load any labels
        // now, after modifying GenericLoader to have this new behavior
        // GenericLoader::load(config.inputFile.c_str(), inputData, 0, n, config.batchSize);
        loader->load(inputData, 0, n, config.batchSize);
    }
    while(more) {
        // no point in forwarding through all, so forward through each, one by one
        if(config.outputLayer < 0 || config.outputLayer > net->getNumLayers()) {
            throw runtime_error("outputLayer should be the layer number of one of the layers in the network");
        }
        dynamic_cast<InputLayer *>(net->getLayer(0))->in(inputData);
        for(int layerId = 0; layerId <= config.outputLayer; layerId++) {
            StatefulTimer::setPrefix("layer" + toString(layerId) + " ");
            net->getLayer(layerId)->forward();
            StatefulTimer::setPrefix("");
        }

        if(!config.writeLabels) {
            if(config.outputFormat == "text") {
                float const*output = net->getLayer(config.outputLayer)->getOutput();
                const int numFields = net->getLayer(config.outputLayer)->getOutputCubeSize();
                //cout << "writing as text n=" << n << " N=" << N << " batchsize=" << config.batchSize <<
                //     " numFields=" << numFields << endl;
                for(int i = 0; i < config.batchSize && (N==-1 || n + i < N); i++) {
                    for(int f = 0; f < numFields; f++) {
                        if(f > 0) {
                            *outFile << " ";
                        }
                        *outFile << output[ i * numFields + f ];
                        //cout << "writing " << output[ i * numFields + f ] << endl;
                    }
                    *outFile << "\n";
                }
            } else {
                outFile->write(reinterpret_cast<const char *>(net->getOutput()), net->getOutputNumElements() * 4 * config.batchSize);
            }
        } else {
            SoftMaxLayer *softMaxLayer = dynamic_cast< SoftMaxLayer *>(net->getLayer(config.outputLayer) );
            if(softMaxLayer == 0) {
                cout << "must choose softmaxlayer, if want to output labels" << endl;
                return;
            }
            softMaxLayer->getLabels(labels);
            if(config.outputFormat == "text") {
                for(int i = 0; i < config.batchSize && (N==-1 || n + i < N); i++) {
                    *outFile << labels[i] << "\n";
                }
            } else {
                int numToWrite = config.batchSize;
                if(N - n < numToWrite) {
                    numToWrite = N - n;
                }
                outFile->write(reinterpret_cast< char * >(labels), numToWrite * 4l);
            }
//            outFile->flush();
        }
        outFile->flush();
        n += config.batchSize;
        if(config.inputFile == "") {
            cin.read(reinterpret_cast< char * >(inputData), inputCubeSize * config.batchSize * 4l);
            more = !cin.eof();
        } else {
            if(n < N) {
                // GenericLoader::load(config.inputFile.c_str(), inputData, 0, n, config.batchSize);
                loader->load(inputData, 0, n, config.batchSize);
            } else {
                more = false;
            }
        }
    }
    if(config.outputFile != "") {
        delete outFile;
    }
    if(loader != NULL) delete loader;

    delete[] inputData;
    delete[] labels;
    delete weightsInitializer;
    delete net;
    delete cl;
}

void printUsage(char *argv[], Config config) {
    cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
    cout << endl;
    cout << "Possible key=value pairs:" << endl;
    /* [[[cog
        cog.outl('// generated using cog:')
        cog.outl('cout << "public api, shouldnt change within major version:" << endl;')
        for option in options:
            name = option['name']
            description = option['description']
            if 'ispublicapi' in option and option['ispublicapi']:
                cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
        cog.outl('cout << "" << endl; ')
        cog.outl('cout << "unstable, might change within major version:" << endl; ')
        for option in options:
            if 'ispublicapi' not in option or not option['ispublicapi']:
                name = option['name']
                description = option['description']
                cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
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
    cout << "    outputlayer=[layer to write output from, default -1 means: last layer] (" << config.outputLayer << ")" << endl;
    cout << "    writelabels=[write integer labels, instead of probabilities etc (default 0)] (" << config.writeLabels << ")" << endl;
    cout << "    outputformat=[output format [binary|text]] (" << config.outputFormat << ")" << endl;
    // [[[end]]]
}

int main(int argc, char *argv[]) {
    Config config;
    if(argc == 2 && (string(argv[1]) == "--help" || string(argv[1]) == "--?" || string(argv[1]) == "-?" || string(argv[1]) == "-h") ) {
        printUsage(argv, config);
    }
    for(int i = 1; i < argc; i++) {
        vector<string> splitkeyval = split(argv[i], "=");
        if(splitkeyval.size() != 2) {
          cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
          exit(1);
        } else {
            string key = splitkeyval[0];
            string value = splitkeyval[1];
//            cout << "key [" << key << "]" << endl;
            /* [[[cog
                cog.outl('// generated using cog:')
                cog.outl('if(false) {')
                for option in options:
                    name = option['name']
                    type = option['type']
                    cog.outl('} else if(key == "' + name.lower() + '") {')
                    converter = '';
                    if type == 'int':
                        converter = 'atoi';
                    elif type == 'float':
                        converter = 'atof';
                    cog.outl('    config.' + name + ' = ' + converter + '(value);')
            */// ]]]
            // generated using cog:
            if(false) {
            } else if(key == "gpuindex") {
                config.gpuIndex = atoi(value);
            } else if(key == "weightsfile") {
                config.weightsFile = (value);
            } else if(key == "batchsize") {
                config.batchSize = atoi(value);
            } else if(key == "inputfile") {
                config.inputFile = (value);
            } else if(key == "outputfile") {
                config.outputFile = (value);
            } else if(key == "outputlayer") {
                config.outputLayer = atoi(value);
            } else if(key == "writelabels") {
                config.writeLabels = atoi(value);
            } else if(key == "outputformat") {
                config.outputFormat = (value);
            // [[[end]]]
            } else {
                cout << endl;
                cout << "Error: key '" << key << "' not recognised" << endl;
                cout << endl;
                printUsage(argv, config);
                cout << endl;
                return -1;
            }
        }
    }
    if(config.outputFormat != "text" && config.outputFormat != "binary") {
        cout << endl;
        cout << "outputformat must be 'text' or 'binary'" << endl;
        cout << endl;
        return -1;
    }
    try {
        go(config);
    } catch(runtime_error e) {
        cout << "Something went wrong: " << e.what() << endl;
        return -1;
    }
}


