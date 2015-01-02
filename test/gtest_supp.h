#include "gtest/gtest.h"

#define ASSERT_FLOAT_NEAR( val1, val2, tolerance_as_fraction_val1 ) \
    ASSERT_NEAR( val1, val2, val1 * tolerance_as_fraction_val1 );

#define EXPECT_FLOAT_NEAR( val1, val2, tolerance_as_fraction_val1 ) \
    EXPECT_NEAR( val1, val2, val1 * tolerance_as_fraction_val1 );

//#define ASSERT_FLOAT_NEAR( val1, val2 ) \\
//    ASSERT_NEAR( val1, val2, 0.0001f );

