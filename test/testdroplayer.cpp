// Copyright kikaxa 2016
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "dropout/DropoutLayer.h"
#include "net/NeuralNet.h"
#include "layer/Layer.h"
#include "layer/LayerMakers.h"
#include "net/NeuralNetMould.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/WeightRandomizer.h"

using namespace std;

namespace testdroplayer {

TEST( testdroplayer, simple_exception ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = NeuralNet::maker(cl)->imageSize(1)->planes(4)->instance();
    net->addLayer( DropoutMaker::instance() );
    net->addLayer( SquareLossMaker::instance() );
    net->setBatchSize( 1 );

    float *input = new float[net->getLayer(0)->getOutputPlanes()];
    input[0] = 0;
    input[1] = 1;
    input[2] = 3;
    input[3] = 2;

    int outputCubeSize = net->getLayer(0)->getOutputPlanes();
    float *expectedOutput = new float[outputCubeSize];
    WeightRandomizer::randomize(2, expectedOutput, outputCubeSize, -2.0f, 2.0f);

    bool exception = false;
    try {
        net->forward( input );

        float const*output = net->getOutput();
        float sum = 0;
        for( int i = 0; i < net->getLayer(0)->getOutputPlanes(); i++ ) {
            //cout << "output[" << i << "]=" << output[i] << endl;
            sum += output[i];
            EXPECT_LE( 0, output[i] );
            EXPECT_GE( 3, output[i] );
        }
        EXPECT_FLOAT_NEAR( 3, sum );


        net->backward(expectedOutput);

    } catch(runtime_error e) {
        cout << "Something went wrong: " << e.what() << endl;
        exception = true;
    } catch (...) {
        exception = true;
    }

    EXPECT_FLOAT_NEAR( false, exception );

    delete[] input;
    delete[] expectedOutput;
    delete net;
    delete cl;
}

}
