// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <algorithm>

#include "conv/LayerDimensions.h"
#include "test/DeepCLGtestGlobals.h"
#include "test/TestArgsParser.h"

#include "DimFromArgs.h"

using namespace std;

void DimFromArgs::arg( LayerDimensions *p_dim ) {
    TestArgsParser::arg( "inputplanes", &(p_dim->inputPlanes) );
    TestArgsParser::arg( "numinputplanes", &(p_dim->inputPlanes) );
    TestArgsParser::arg( "inputsize", &(p_dim->inputSize) );
    TestArgsParser::arg( "numfilters", &(p_dim->numFilters) );
    TestArgsParser::arg( "filtersize", &(p_dim->filterSize) );
    TestArgsParser::arg( "padzeros", &(p_dim->padZeros) );
    TestArgsParser::arg( "biased", &(p_dim->biased) );
//    cout << "DimFromArgs::arg() " << *p_dim << endl;
}


