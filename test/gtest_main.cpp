// Copyright Hugh Perkins 2013,2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "gtest/gtest.h"

#include "test/GtestGlobals.h"

using namespace std;

string cmdline;

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    GtestGlobals::instance()->argc = argc;
    GtestGlobals::instance()->argv = argv;
    return RUN_ALL_TESTS();
}

