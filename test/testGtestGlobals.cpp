// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

#include "DeepCLGtestGlobals.h"

using namespace std;

TEST( testDeepCLGtestGlobals, basic ) {
    cout << "There are " << DeepCLGtestGlobals::instance()->argc << " parameters: " << endl;
    for( int i = 0; i < DeepCLGtestGlobals::instance()->argc; i++ ) {
        cout << "   argv[" << i << "]=" << DeepCLGtestGlobals::instance()->argv[i] << endl;
    }
}

