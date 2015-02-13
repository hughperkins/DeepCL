// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "stringhelper.h"

#include "gtest/gtest.h"

#include "test/GtestGlobals.h"

using namespace std;

string cmdline;

GTEST_API_ int main(int argc, char **argv) {
    // add filter on slow by default
    bool deleteargv = false;
    if( argc == 1 ) {
        char **newargv = new char *[argc + 1];
        int newargc = argc + 1;
        for( int i = 0; i < argc; i++ ) {
            char *newarg = new char[strlen( argv[i]) + 1 ];
            newargv[i] = newarg;
            strcpy_safe( newargv[i], argv[i], 255 );
        }
        string slowfilter = "--gtest_filter=-SLOW*";
        newargv[argc] = new char[ slowfilter.length() + 1 ];
        strcpy_safe( newargv[argc], slowfilter.c_str(), 255 );
        argv = newargv;
        argc = newargc;
        deleteargv = true;
    }
//    cout << "argc " << argc << endl;
//    cout << "args:";
//    for( int i = 0; i < argc; i++ ) {
//        cout << " " << argv[i];
//    }
//    cout << endl;
    testing::InitGoogleTest(&argc, argv);
    GtestGlobals::instance()->argc = argc;
    GtestGlobals::instance()->argv = argv;
    int retValue = RUN_ALL_TESTS();
    if( deleteargv ) {
        for( int i = 0; i < argc; i++ ) {
            delete[] argv[i];
        }
        delete[] argv;
    }
    return retValue;
}

