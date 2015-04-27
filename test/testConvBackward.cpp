// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "OpenCLHelper.h"

#include "BackpropErrorsv2.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/TestArgsParser.h"
#include "test/WeightRandomizer.h"

using namespace std;

TEST( testdropoutbackprop, basic ) {
    int batchSize = 32;

    int inputPlanes = 2;
    int inputSize = 3;
    int numFilters = 2;
    int filterSize = 2;

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    LayerDimensions dim( numInPlanes, inputSize, numFilters, filterSize,
            padZeros == 0, false );

    BackpropErrorsv2 backpropErrorsv2* = BackpropErrorsv2::instanceForTest( cl, dim );
    float gradOutput[] = {
        3, 5,-2.7f,
        2, -9, 2.1f,
        0, -1.1f, 3.5f
    };
    int inputTotalSize = dropoutBackprop->getInputSize( batchSize );
    EXPECT_FLOAT_NEAR( batchSize * imageSize * imageSize, inputTotalSize );
    float *errorsForUpstream = new float[ inputTotalSize ];

    dropoutBackprop->backward( batchSize, mask, errors, errorsForUpstream );

    EXPECT_FLOAT_NEAR( 3, errorsForUpstream[0] );
    EXPECT_FLOAT_NEAR( 5, errorsForUpstream[1] );
    EXPECT_FLOAT_NEAR( 0, errorsForUpstream[2] );

    EXPECT_FLOAT_NEAR( 0, errorsForUpstream[3] );
    EXPECT_FLOAT_NEAR( -9, errorsForUpstream[4] );
    EXPECT_FLOAT_NEAR( 2.1f, errorsForUpstream[5] );

    EXPECT_FLOAT_NEAR( 0, errorsForUpstream[6] );
    EXPECT_FLOAT_NEAR( 0, errorsForUpstream[7] );
    EXPECT_FLOAT_NEAR( 3.5f, errorsForUpstream[8] );
//    for( int i = 0; i < 16; i++ ) {
//        EXPECT_FLOAT_NEAR( expectedErrorsForUpstream[i], errorsForUpstream[i] );
//    }

    delete dropoutBackprop;
    delete[] errorsForUpstream;
    delete cl;
}
