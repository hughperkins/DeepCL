// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <iomanip>
#include <algorithm>

#include "DeepCL.h"
#include "conv/Forward.h"

#include "gtest/gtest.h"

using namespace std;

#include "test/gtest_supp.h"

TEST(testDeepCL, basic) {
    DeepCL *cl = DeepCL::createForFirstGpuOtherwiseCpu();

    int batchSize = 2;
    int numInPlanes = 1; int imageSize = 2;
    int numOutPlanes = 2; int filterWidth = 2;
    int padZeros = 0;
    float data[] = { 0, 0, 
                      0.5f, 0.5f,

                        13, 17,
                       -19, 2.3f,
};
    float filter1[] = { 0, 0,
                        -0.5f, 0.5f,

                        0.2f, 0.3f, 
                         0.7f, -1.1f,
 };
    int resultSize = 4;
    float expectedOutput[] = {
        -0.5f * 0.5f + 0.5f * 0.5f,
        0.7f * 0.5f -1.1f * 0.5f,
        (-0.5f) * (-19) + 0.5f * 2.3f,
        0.2f*13 + 0.3f* 17 + 0.7f *(-19) -1.1f * 2.3f 
    };
    cout << "expected number of output: " << resultSize << endl;
//    int outputSize = 0;
    for( int i = 1; i <= 4; i++ ) {
        Forward *forward = Forward::instanceSpecific( 3, cl,
            LayerDimensions( numInPlanes, imageSize, numOutPlanes, filterWidth,
            padZeros == 1, false ) );
        float *output = new float[forward->getOutputTotalSize(batchSize)];
        forward->forward( batchSize, data, filter1, 0, output );  
        for( int result = 0; result < resultSize; result++ ) {
            ASSERT_EQ( expectedOutput[result], output[result] );
        }
        delete forward;
        delete[] output;
    }

    delete cl;
}

