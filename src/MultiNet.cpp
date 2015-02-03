// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NormalizationHelper.h"
#include "NeuralNet.h"

#include "MultiNet.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

MultiNet::MultiNet( int numNets, NeuralNet *model ) {
    nets.push_back( model );
    for( int i = 0; i < numNets; i++ ) {
        nets.push_back( model->clone() );
    }
}


