// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/TestArgsParser.h"
#include "GtestGlobals.h"

using namespace std;

TEST( testArgsParser, basic ) {
    int myvalue = 3;
    TestArgsParser argsParser;
    argsParser._arg( "myvalue", &myvalue );
    argsParser._go();
    cout << "myvalue: " << myvalue << endl;
}

TEST( testArgsParser, static ) {
    GtestGlobals::instance()->argc = 2;
    GtestGlobals::instance()->argv = new char *[2];
    GtestGlobals::instance()->argv[0] = new char[255];    
    GtestGlobals::instance()->argv[1] = new char[255];    
    strcpy_safe( GtestGlobals::instance()->argv[0], "./unittests", 20 );
    strcpy_safe( GtestGlobals::instance()->argv[1], "myvalue=5", 20 );

    int myvalue = 3;
    TestArgsParser::arg( "myvalue", &myvalue );
    TestArgsParser::go();
    cout << "myvalue: " << myvalue << endl;

    myvalue = 3;
    bool threw = false;
    try {
        TestArgsParser::go();
    } catch( runtime_error e ) {
        threw = true;
    }
    EXPECT_EQ( true, threw );
//    cout << "myvalue: " << myvalue << endl;

    myvalue = 3;
    TestArgsParser::arg( "myvalue", &myvalue );
    TestArgsParser::go();
    cout << "myvalue: " << myvalue << endl;
}

