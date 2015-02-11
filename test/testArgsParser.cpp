// Copyright Hugh Perkins 2013,2014 hughperkins at gmail
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
    TestArgsParser argsParser(GtestGlobals::instance()->argc, GtestGlobals::instance()->argv );
    argsParser.arg( "myvalue", &myvalue );
    argsParser.go();
    cout << "myvalue: " << myvalue << endl;
}

