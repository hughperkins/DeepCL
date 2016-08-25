// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <string>

#include "net/NeuralNet.h"
#include "layer/LayerMakers.h"
#include "util/stringhelper.h"
#include "netdef/NetdefToNet.h"
#include "activate/ActivationFunction.h"
#include "weights/WeightsInitializer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

// string is structured like:
// prefix-nn*(inner)-postfix
// or:
// prefix-nn*inner-postfix
STATIC std::string expandMultipliers(std::string netdef) {
    size_t starPos = netdef.find("*");
    if(starPos != string::npos) {
        size_t prefixEnd = netdef.rfind("-", starPos);
        string prefix = "";
        string nnString = "";
        if(prefixEnd == string::npos) {
            prefixEnd = -1;
            nnString = netdef.substr(0, starPos);
        } else {
            prefixEnd--;
            prefix = netdef.substr(0, prefixEnd + 1);
            cout << "prefix: [" << prefix << "]" << endl;
            nnString = netdef.substr(prefixEnd + 2, starPos - prefixEnd - 2);
        }
        cout << "nnString: [" << nnString << "]" << endl;
        int repeatNum = atoi(nnString);
        cout << "repeatNum " << repeatNum << endl;
        string remainderString = netdef.substr(starPos + 1);
        cout << "remainderString [" << remainderString << "]" << endl;
        string inner = "";
        string postfix = "";
        if(remainderString.substr(0, 1) == "(") {
            // need to find other ')', assume not nested for now...
            size_t rhBracket = remainderString.find(")");
            if(rhBracket == string::npos) {
                throw runtime_error("matching bracket not found in " + remainderString);
            }
            inner = remainderString.substr(1, rhBracket - 1);
            cout << "inner [" << inner << "]" << endl;
            string newRemainder = remainderString.substr(rhBracket + 1);
            cout << "newRemainder [" << newRemainder << "]" << endl;
            if(newRemainder != "") {
                if(newRemainder[0] != '-') {
                    throw runtime_error("expect '-' after ')' in " + remainderString);
                }
                postfix = newRemainder.substr(1);
                cout << "postfix [" << postfix << "]" << endl;
            }
        } else {
            size_t innerEnd = remainderString.find("-");
            if(innerEnd == string::npos) {
                innerEnd = remainderString.length();
            } else {
//                innerEnd;
                postfix = remainderString.substr(innerEnd + 1);
                cout << "postfix [" << postfix << "]" << endl;
            }
            inner = remainderString.substr(0, innerEnd);
            cout << "inner [" << inner << "]" << endl;
//            if(remainderString.find("-") != string::npos) {
//                sectionEndPos = remainderString.find("-");
//            }
        }
//        return "";
        // if remainderString starts with (, then repeat up to next)
        // otherwise, repeat up to next -
//        int sectionEndPos = remainderString.length();
//        remainderString = 
        string newString = prefix;
        for(int i = 0; i < repeatNum; i++) {
            if(newString != "") {
                newString += "-";
            }
            newString += expandMultipliers(inner);
        }
        if(postfix != "") {
            newString += "-" + expandMultipliers(postfix);
        }
        cout << "multiplied string: " << newString << endl;
        return newString;
    } else {
        return netdef;
    }    
}

STATIC bool NetdefToNet::parseSubstring(WeightsInitializer *weightsInitializer, NeuralNet *net, std::string substring, bool isLast) {
//    cout << "substring [" << substring << "]" << endl;
    vector<string>splitLayerDef = split(substring, "{");
    string baseLayerDef = splitLayerDef[0];
//         optionsDef = "";
    vector<string> splitOptionsDef;
//    cout << "splitlayerdef.size() " << splitLayerDef.size() << endl;
    if(splitLayerDef.size() == 2) {
        string  optionsDef = split(splitLayerDef[1], "}")[0];
//        cout << "optionsDef [" << optionsDef << "]" << endl;
        splitOptionsDef = split(optionsDef, ",");
    }
    if(baseLayerDef.find("c") != string::npos) {
        vector<string> splitConvDef = split(baseLayerDef, "c");
        int numFilters = atoi(splitConvDef[0]);
        vector<string> splitConvDef1 = split(splitConvDef[1], "z");
        int filterSize = atoi(splitConvDef1[0]);
        int skip = 0;
        ActivationFunction *fn = 0;
        bool padZeros = splitConvDef1.size() == 2 ? true : false;

        for(int i = 0; i < (int)splitOptionsDef.size(); i++) {
            string optionDef = splitOptionsDef[i];
//            cout << "optionDef [" << optionDef << "]" << endl;
            vector<string> splitOptionDef = split(optionDef, "=");
            string optionName = splitOptionDef[0];
            if(splitOptionDef.size() == 2) {
                string optionValue = splitOptionDef[1];
                if(optionName == "skip") {
                    skip = atoi(optionValue);
                    cout << "got skip: " << skip << endl;
                }
            } else if(splitOptionDef.size() == 1) {
                if(optionName == "tanh") {
                    fn = new TanhActivation();
                } else if(optionName == "scaledtanh") {
                    fn = new ScaledTanhActivation();
                } else if(optionName == "sigmoid") {
                    fn = new SigmoidActivation();
                } else if(optionName == "relu") {
                    fn = new ReluActivation();
                } else if(optionName == "elu") {
                    fn = new EluActivation();
                } else if(optionName == "linear") {
                    fn = new LinearActivation();
                } else if(optionName == "padzeros" || optionName == "z") {
                    padZeros = true;
                } else {
                    cout << "Error: unknown subkey: [" << optionName << "]" << endl;
                    return false;
                }
            } else {
                cout << "Error: unknown subkey: [" << optionName << "]" << endl;
                return false;
            }
        }
        net->addLayer(ConvolutionalMaker::instance()->numFilters(numFilters)->filterSize(filterSize)->padZeros(padZeros)->biased()->weightsInitializer(weightsInitializer) );
        if(fn != 0) {
            net->addLayer(ActivationMaker::instance()->fn(fn) );
        }
    } else if(baseLayerDef.find("mp") != string::npos) {
        vector<string> splitPoolDef = split(baseLayerDef, "mp");
        int poolingSize = atoi(splitPoolDef[1]);
        net->addLayer(PoolingMaker::instance()->poolingSize(poolingSize));
    } else if(baseLayerDef.find("drop") != string::npos) {
        net->addLayer(DropoutMaker::instance()->dropRatio(0.5f));
    } else if(baseLayerDef.find("relu") != string::npos) {
        net->addLayer(ActivationMaker::instance()->relu());
    } else if(baseLayerDef.find("elu") != string::npos) {
        net->addLayer(ActivationMaker::instance()->elu());
    } else if(baseLayerDef.find("tanh") != string::npos) {
        net->addLayer(ActivationMaker::instance()->tanh());
    } else if(baseLayerDef.find("sigmoid") != string::npos) {
        net->addLayer(ActivationMaker::instance()->sigmoid());
    } else if(baseLayerDef.find("linear") != string::npos) {
        net->addLayer(ActivationMaker::instance()->linear()); // kind of pointless nop, but useful for testing
    } else if(baseLayerDef.find("rp") != string::npos) {
        int patchSize = atoi(split(baseLayerDef, "rp")[1]);
        net->addLayer(RandomPatchesMaker::instance()->patchSize(patchSize) );
    } else if(baseLayerDef.find("rt") != string::npos) {
        int translateSize = atoi(split(baseLayerDef, "rt")[1]);
        net->addLayer(RandomTranslationsMaker::instance()->translateSize(translateSize) );
    } else if(baseLayerDef.find("n") != string::npos) {
        vector<string> fullDef = split(baseLayerDef, "n");
        int numPlanes = atoi(fullDef[0]);
        ActivationFunction *fn = 0;
//        if(isLast) {
//            fn = new LinearActivation();
//        }
//        int padZeros = 0;
        bool biased = true;
        for(int i = 0; i < (int)splitOptionsDef.size(); i++) {
            string optionDef = splitOptionsDef[i];
//                cout << "optionDef: " << optionDef << endl;
            vector<string> splitOptionDef = split(optionDef, "=");
            string optionName = splitOptionDef[0];
            if(splitOptionDef.size() == 1) {
                if(optionName == "tanh") {
                    fn = new TanhActivation();
                } else if(optionName == "scaledtanh") {
                    fn = new ScaledTanhActivation();
                } else if(optionName == "sigmoid") {
                    fn = new SigmoidActivation();
                } else if(optionName == "relu") {
                    fn = new ReluActivation();
                } else if(optionName == "nobias") {
                    biased = false;
                } else if(optionName == "linear") {
                    fn = new LinearActivation();
                } else {
                    cout << "Error: unknown subkey: [" << splitOptionsDef[i] << "]" << endl;
                    return false;
                }
            } else {
                cout << "Error: unknown subkey: [" << splitOptionsDef[i] << "]" << endl;
                return false;
            }
        }
        if(isLast && fn != 0) {
            cout << "Last fullyconnectedlayer must be linear (because softmax is the 'activationlayer' for this layer)" << endl;
            return false;
        }
        net->addLayer(FullyConnectedMaker::instance()->numPlanes(numPlanes)->imageSize(1)->biased(biased)->weightsInitializer(weightsInitializer) );
        if(fn != 0) {
            net->addLayer(ActivationMaker::instance()->fn(fn) );
        }
    } else {
        cout << "network definition " << baseLayerDef << " not recognised" << endl;
        return false;
    }
    return true;
}

PUBLICAPI STATIC bool NetdefToNet::createNetFromNetdef(NeuralNet *net, std::string netdef) {
    OriginalInitializer originalInitializer;
    return createNetFromNetdef(net, netdef, &originalInitializer);
}
PUBLICAPI STATIC bool NetdefToNet::createNetFromNetdefCharStar(NeuralNet *net, const char *netdef) {
    OriginalInitializer originalInitializer;
    return createNetFromNetdef(net, netdef, &originalInitializer);
}

STATIC bool NetdefToNet::createNetFromNetdef(NeuralNet *net, std::string netdef, WeightsInitializer *weightsInitializer) {
    string netDefLower = toLower(netdef);
//    cout << "netDefLower [" << netDefLower << "]" << endl;
    try {
        netDefLower = expandMultipliers(netDefLower);
    } catch(runtime_error &e) {
        cout << e.what() << endl;
        return false;
    }
//    cout << "netDefLower [" << netDefLower << "]" << endl;
    vector<string> splitNetDef = split(netDefLower, "-");
    if(netdef != "") {
        for(int i = 0; i < (int)splitNetDef.size(); i++) {
            string thisLayerDef = splitNetDef[i];
//            cout << "thisLayerDef [" << thisLayerDef << "]" << endl;
            if(!parseSubstring(weightsInitializer, net, thisLayerDef, i == (int)splitNetDef.size() - 1) ) {
                return false;
            }
        }
    }
    net->addLayer(SoftMaxMaker::instance());
    return true;
}