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

TEST( teststringhelper, replace ) {
    string mystring = "hellonewworld";
    EXPECT_EQ( "hellocoolworld", replace( mystring, "new", "cool" ) );
}

TEST( teststringhelper, replaceglobal ) {
    string mystring = "";
    mystring = "hello world";
    EXPECT_EQ( "one world", replaceGlobal( mystring, "hello", "one" ) );

    mystring = "hello hello";
    EXPECT_EQ( "one one", replaceGlobal( mystring, "hello", "one" ) );

    mystring = "hello hello hello";
    EXPECT_EQ( "one one one", replaceGlobal( mystring, "hello", "one" ) );

    mystring = "hellohellohello";
    EXPECT_EQ( "oneoneone", replaceGlobal( mystring, "hello", "one" ) );

    mystring = "hellonewwohellorldhellohellohello";
    EXPECT_EQ( "onenewwoonerldoneoneone", replaceGlobal( mystring, "hello", "one" ) );
}

TEST( teststringhelper, strcpy_safe ) {
    char const*source = "hello123";
    char dest[1024];
    memset( dest, 99, 1024 ); // ie, not zero :-)
    strcpy_safe( dest, source, 100 );
    string target = string(dest);
    EXPECT_EQ( "hello123", target );
    EXPECT_EQ( 0, dest[8] );
    EXPECT_EQ( 99, dest[9] );
    EXPECT_EQ( '3', dest[7] );

    memset( dest, 99, 1024 ); // ie, not zero :-)
    strcpy_safe( dest, source, 3 );
    target = string(dest);
    EXPECT_EQ( "hel", target );
    EXPECT_EQ( 0, dest[3] );
    EXPECT_EQ( 99, dest[4] );
    EXPECT_EQ( 'l', dest[2] );
}

