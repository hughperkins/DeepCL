// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NeuralNet.h"

#include "NetdefToNet.h"

using namespace std;

bool NetdefToNet::createNetFromNetdef( NeuralNet *net, std::string netdef ) {
    string netDefLower = toLower( netdef );
    vector<string> splitNetDef = split( netDefLower, "-" );
    for( int i = 0; i < splitNetDef.size(); i++ ) {
        string thisLayerDef = splitNetDef[i];
        vector<string>splitLayerDef = split( thisLayerDef, "{" );
        string baseLayerDef = splitLayerDef[0];
        string optionsDef = "";
        vector<string> splitOptionsDef;
        if( splitLayerDef.size() == 2 ) {
            optionsDef = split( splitLayerDef[1], "}" )[0];
            splitOptionsDef = split( optionsDef, "," );
        }
        cout << "optionsDef: " << optionsDef << endl;
        if( baseLayerDef.find("c") != string::npos ) {
            vector<string> splitConvDef = split( baseLayerDef, "c" );
            int numFilters = atoi( splitConvDef[0] );
            int filterSize = atoi( splitConvDef[1] );
            int skip = 0;
            ActivationFunction *fn = new ReluActivation();
            int padZeros = 0;
            for( int i = 0; i < splitOptionsDef.size(); i++ ) {
                string optionDef = splitOptionsDef[i];
                cout << "optionDef: " << optionDef << endl;
                vector<string> splitOptionDef = split( optionDef, "=");
                string optionName = splitOptionDef[0];
                if( splitOptionDef.size() == 2 ) {
                    string optionValue = splitOptionDef[1];
                    if( optionName == "skip" ) {
                        skip = atoi( optionValue );
                        cout << "got skip: " << skip << endl;
                    }
                } else if( splitOptionDef.size() == 1 ) {
                    if( optionName == "tanh" ) {
                        fn = new TanhActivation();
                    } else if( optionName == "scaledtanh" ) {
                        fn = new ScaledTanhActivation();
                    } else if( optionName == "sigmoid" ) {
                        fn = new SigmoidActivation();
                    } else if( optionName == "relu" ) {
                        fn = new ReluActivation();
                    } else if( optionName == "linear" ) {
                        fn = new LinearActivation();
                    } else if( optionName == "padzeros" ) {
                        padZeros = 1;
                    } else {
                        cout << "Error: unknown subkey: [" << splitOptionsDef[i] << "]" << endl;
                        return false;
                    }
                } else {
                    cout << "Error: unknown subkey: [" << splitOptionsDef[i] << "]" << endl;
                    return false;
                }
            }
            net->convolutionalMaker()->numFilters(numFilters)->filterSize(filterSize)->fn( fn )->padZeros( padZeros )->biased()->insert();
        } else if( baseLayerDef.find("mp") != string::npos ) {
            vector<string> splitPoolDef = split( baseLayerDef, "mp" );
            int poolingSize = atoi( splitPoolDef[1] );
            net->poolingMaker()->poolingSize(poolingSize)->insert();
        } else if( baseLayerDef.find("rp") != string::npos ) {
            int patchSize = atoi( split( baseLayerDef, "rp" )[1] );
            net->randomPatchesMaker()->patchSize( patchSize )->insert();
        } else if( baseLayerDef.find("rt") != string::npos ) {
            int translateSize = atoi( split( baseLayerDef, "rt" )[1] );
            net->randomTranslationsMaker()->translateSize( translateSize )->insert();
        } else if( baseLayerDef.find("n") != string::npos ) {
            vector<string> fullDef = split( baseLayerDef, "n" );
            int numPlanes = atoi( fullDef[0] );
            if( i == splitNetDef.size() - 1 ) {
                net->fullyConnectedMaker()->numPlanes(numPlanes)->boardSize(1)->linear()->biased()->insert();
            } else {
                net->fullyConnectedMaker()->numPlanes(numPlanes)->boardSize(1)->tanh()->biased()->insert();
            }
        } else {
            cout << "network definition " << baseLayerDef << " not recognised" << endl;
            return false;
        }
    }
    net->softMaxLossMaker()->insert();
    return true;
}

