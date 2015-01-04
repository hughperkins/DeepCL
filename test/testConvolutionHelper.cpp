#include "gtest/gtest.h"
#include "test/gtest_supp.h"

#include "ConvolutionHelper.h"

using namespace std;

TEST( testConvolutionHelper, one ) {
    // need to provide 2 5-dimension inputs... , and then the expected 5-dimension output...
    float *in1 = new float[1];
    float *in2 = new float[1];
    float *out = new float[1];
    int *din1 = new int[5];
    int *din2 = new int[5];
    int *dout = new int[5];
    int *zeroPad = new int[5];
    arraySet( 5, din1, 1 );
    arraySet( 5, din2, 1 );
    arraySet( 5, dout, 1 );
    arraySet( 5, zeroPad, 0 );
    in1[0] = 2;
    in2[1] = 3;
    ConvolutionHelper::convolveCpu( 5, din1, din2, dout, zeroPad, in1, in2, out );
    EXPECT_EQ( 1, dout[0] );
    EXPECT_EQ( 1, dout[1] );
    EXPECT_EQ( 1, dout[2] );
    EXPECT_EQ( 1, dout[3] );
    EXPECT_EQ( 1, dout[4] );
    EXPECT_FLOAT_NEAR( 6, out[0] );
}

