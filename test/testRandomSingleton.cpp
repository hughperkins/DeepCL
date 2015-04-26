// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// going to check:
// - that it works
// - that can replace it with our own mock version

#include <iostream>

#include "MockRandomSingleton.h"

#include "gtest/gtest.h"

using namespace std;

TEST( testRandomSingleton, testMockRandom ) {
    for( int i = 0; i < 10; i++ ) {
        cout << RandomSingleton::uniform() << endl;
    }
    MockRandomSingletonUniforms mock;
    float values[] = { 0.2f, 0.8f,0.3f };
    mock.setValues(3, values);
    RandomSingleton *random = &mock;
    EXPECT_EQ( 0.2f, random->_uniform() );
    EXPECT_EQ( 0.8f, random->_uniform() );
    EXPECT_EQ( 0.3f, random->_uniform() );
}

