// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

#include "GtestGlobals.h"

using namespace std;

TEST( testGtestGlobals, basic ) {
    cout << "There are " << GtestGlobals::instance()->argc << " parameters: " << endl;
    for( int i = 0; i < GtestGlobals::instance()->argc; i++ ) {
        cout << "   argv[" << i << "]=" << GtestGlobals::instance()->argv[i] << endl;
    }
}

