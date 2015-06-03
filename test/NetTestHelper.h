// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

class NeuralNet;

class NetTestHelper {
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    STATIC void printWeightsAsCode( Layer *layer );
    STATIC void printBiasAsCode( Layer *layer );
    STATIC void printWeightsAsCode(NeuralNet *net);
    STATIC void printBiasAsCode(NeuralNet *net);

    // [[[end]]]
};

