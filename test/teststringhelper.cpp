// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "stringhelper.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

using namespace std;

TEST( teststringhelper, split ) {
    string mystring = "MP10";
    vector<string> splitString = split( mystring, "MP" );
    EXPECT_EQ( "", splitString[0] );
    EXPECT_EQ( "10", splitString[1] );
}

TEST( teststringhelper, split2 ) {
    string mystring = "MP10MPMP54";
    vector<string> splitString = split( mystring, "MP" );
    EXPECT_EQ( "", splitString[0] );
    EXPECT_EQ( "10", splitString[1] );
    EXPECT_EQ( "", splitString[2] );
    EXPECT_EQ( "54", splitString[3] );
}

TEST( teststringhelper, split3 ) {
    string mystring = "42MP10MPMP54";
    vector<string> splitString = split( mystring, "MP" );
    EXPECT_EQ( "42", splitString[0] );
    EXPECT_EQ( "10", splitString[1] );
    EXPECT_EQ( "", splitString[2] );
    EXPECT_EQ( "54", splitString[3] );
}

TEST( teststringhelper, tolower ) {
    string mystring = "3fAfef4FAD";
    EXPECT_EQ( "3fafef4fad", toLower( mystring ) );
}

