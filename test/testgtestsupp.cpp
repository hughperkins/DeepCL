#include "gtest/gtest.h"
#include "test/gtest_supp.h"

TEST(testgtestsupp, main ) {
    float one = 1.314f;
//    float two = 1.315f;
//    float three = 1.319f;
    
//    EXPECT_FLOAT_NEAR( 1.315f, one, 0.001f );
//    EXPECT_FLOAT_NEAR( 1.317f, one, 0.001f );

//    AssertFoo( 1.315f, one );
//    AssertFoo( 1.317f, one );
//    EXPECT_PRED2( floatsNear, 1.314f, one );
//    EXPECT_PRED2( floatsNear, 1.315f, one );
//    EXPECT_PRED2( floatsNear, 1.318f, one );
//    EXPECT_PRED2( floatsNear, 1.328f, one );
//    EXPECT_PRED2( floatsNear, 1.428f, one );

    EXPECT_PRED_FORMAT2( AssertFloatsNear, 1.3141f, one );
    EXPECT_PRED_FORMAT2( AssertFloatsNear, 1.314f, one );
    EXPECT_PRED_FORMAT2( AssertFloatsNear, 1.315f, one );
    EXPECT_PRED_FORMAT2( AssertFloatsNear, 1.318f, one );
    EXPECT_PRED_FORMAT2( AssertFloatsNear, 1.328f, one );
    EXPECT_PRED_FORMAT2( AssertFloatsNear, 1.428f, one );
    EXPECT_PRED_FORMAT2( AssertFloatsNear, 30000, 30100 );
    EXPECT_PRED_FORMAT2( AssertFloatsNear, 30000, 30010 );
    EXPECT_PRED_FORMAT2( AssertFloatsNear, 30000, 30000.1f );
    EXPECT_PRED_FORMAT2( AssertFloatsNear, 0.00030001f, 0.0003000f );
    EXPECT_PRED_FORMAT2( AssertFloatsNear, 0.00030004f, 0.0003000f );
    EXPECT_PRED_FORMAT2( AssertFloatsNear, 0.00030010f, 0.0003000f );

    EXPECT_FLOAT_NEAR( 1.3141f, one );
    EXPECT_FLOAT_NEAR( 1.314f, one );
    EXPECT_FLOAT_NEAR( 1.315f, one );
    EXPECT_FLOAT_NEAR( 1.318f, one );
    EXPECT_FLOAT_NEAR( 1.328f, one );
    EXPECT_FLOAT_NEAR( 1.428f, one );
    EXPECT_FLOAT_NEAR( 0.00030010f, 0.0003000f );
}


