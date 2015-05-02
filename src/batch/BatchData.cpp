// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "batch/BatchData.h"
#include "net/Trainable.h"

using namespace std;

InputData *InputData::instance( Trainable *net, float *inputs ) {
    int inputCubeSize = net->getInputCubeSize();
    return new InputData( inputCubeSize, inputs );
}
ExpectedData *ExpectedData::instance( Trainable *net, float *expectedOutputs ) {
    int outputCubeSize = net->getOutputCubeSize();
    return new ExpectedData( outputCubeSize, expectedOutputs );
}
LabeledData *LabeledData::instance( int *labels ) {
    return new LabeledData( labels );
}

