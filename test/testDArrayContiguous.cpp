#include "DArrayContiguous.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

TEST( testDArrayContiguous, one ) {
    DArrayContiguous<float> one(2);
    one(0) = 3;
    one(1) = 5.2f;
    EXPECT_FLOAT_NEAR( 3, one(0) );
    EXPECT_FLOAT_NEAR( 5.2f, one(1) );
    one(1)++;
    EXPECT_FLOAT_NEAR( 3, one(0) );
    EXPECT_FLOAT_NEAR( 6.2f, one(1) );
    float *jagged1 = one.getJagged1d();
    jagged1[0] = 5.1f;
    EXPECT_FLOAT_NEAR( 5.1f, one(0) );

    DArrayContiguous<float> two(2,3 );
    two(0,1) = 5.2f;
    two(1,2) = 3.4f;
    EXPECT_FLOAT_NEAR( 5.2f, two(0,1) );
    EXPECT_FLOAT_NEAR( 3.4f, two(1,2) );
    float **jagged2 = one.getJagged2d();
    EXPECT_FLOAT_NEAR( 5.2f, jagged2[0][1] );
    EXPECT_FLOAT_NEAR( 3.4f, jagged2[1][2] );
}

