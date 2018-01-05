// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "clmath/CLMathWrapper.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/WeightRandomizer.h"

using namespace std;

TEST(testCLMathWrapper, assign) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    float bdat[] = { 4,2.1f, 5,3,9.2f };
    CLWrapper *a_ = cl->wrap(5,adat);
    CLWrapper *b_ = cl->wrap(5,bdat);
    a_->copyToDevice();
    b_->copyToDevice();

    CLMathWrapper a(a_);
    CLMathWrapper b(b_);
    a = b;
    a_->copyToHost();

    for(int i = 0; i < 5; i++) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR(2.1f, adat[1]);
    EXPECT_FLOAT_NEAR(9.2f, adat[4]);

//    delete a;
//    delete b;
    delete a_;
    delete b_;
    delete cl;
}

TEST(testCLMathWrapper, assignScalar) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    CLWrapper *a_ = cl->wrap(5,adat);
    a_->copyToDevice();

    CLMathWrapper a(a_);
    a = 3.4f;
    a_->copyToHost();

    for(int i = 0; i < 5; i++) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR(3.4f, adat[1]);
    EXPECT_FLOAT_NEAR(3.4f, adat[4]);

    delete a_;
    delete cl;
}

TEST(testCLMathWrapper, addinplace) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    float bdat[] = { 4,2.1f, 5,3,9.2f };
    CLWrapper *a_ = cl->wrap(5,adat);
    CLWrapper *b_ = cl->wrap(5,bdat);
    a_->copyToDevice();
    b_->copyToDevice();

    CLMathWrapper a(a_);
    CLMathWrapper b(b_);
    a += b;
    a_->copyToHost();

    for(int i = 0; i < 5; i++) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR(5.0f, adat[0]);
    EXPECT_FLOAT_NEAR(5.1f, adat[1]);
    EXPECT_FLOAT_NEAR(2.5f + 9.2f, adat[4]);

//    delete a;
//    delete b;
    delete a_;
    delete b_;
    delete cl;
}

TEST(testCLMathWrapper, multiplyinplace) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
//    float bdat[] = { 4,2.1f, 5,3,9.2f };
    CLWrapper *a_ = cl->wrap(5,adat);
//    CLWrapper *b_ = cl->wrap(5,bdat);
    a_->copyToDevice();
//    b_->copyToDevice();

    CLMathWrapper a(a_);
//    CLMathWrapper b(b_);
//    a += b;
    a *= 1.5f;
    a_->copyToHost();

    for(int i = 0; i < 5; i++) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR(1.5f, adat[0]);
    EXPECT_FLOAT_NEAR(4.5f, adat[1]);
    EXPECT_FLOAT_NEAR(3.75f, adat[4]);

//    delete a;
//    delete b;
    delete a_;
//    delete b_;
    delete cl;
}

TEST(testCLMathWrapper, addscalar) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    CLWrapper *a_ = cl->wrap(5,adat);
    a_->copyToDevice();

    CLMathWrapper a(a_);
    a += 1.5f;
    a_->copyToHost();

    for(int i = 0; i < 5; i++) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR(2.5f, adat[0]);
    EXPECT_FLOAT_NEAR(4.5f, adat[1]);
    EXPECT_FLOAT_NEAR(4.0f, adat[4]);

    delete a_;
    delete cl;
}

TEST(testCLMathWrapper, sqrt) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    CLWrapper *a_ = cl->wrap(5,adat);
    a_->copyToDevice();

    CLMathWrapper a(a_);
    a.sqrt();
    a_->copyToHost();

    for(int i = 0; i < 5; i++) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR(1, adat[0]);
    EXPECT_FLOAT_NEAR(1.73205f, adat[1]);
    EXPECT_FLOAT_NEAR(3, adat[2]);
    EXPECT_FLOAT_NEAR(sqrt(2.5f), adat[4]);

    delete a_;
    delete cl;
}

TEST(testCLMathWrapper, squared) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    CLWrapper *a_ = cl->wrap(5,adat);
    a_->copyToDevice();

    CLMathWrapper a(a_);
    a.squared();
    a_->copyToHost();

    for(int i = 0; i < 5; i++) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR(1, adat[0]);
    EXPECT_FLOAT_NEAR(9, adat[1]);
    EXPECT_FLOAT_NEAR(81, adat[2]);
    EXPECT_FLOAT_NEAR(2.5f * 2.5f, adat[4]);

    delete a_;
    delete cl;
}

TEST(testCLMathWrapper, inverse) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    CLWrapper *a_ = cl->wrap(5,adat);
    a_->copyToDevice();

    CLMathWrapper a(a_);
    a.inv();
    a_->copyToHost();

    for(int i = 0; i < 5; i++) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR(1, adat[0]);
    EXPECT_FLOAT_NEAR(0.333333f, adat[1]);
    EXPECT_FLOAT_NEAR(1.0f / 9.0f, adat[2]);
    EXPECT_FLOAT_NEAR(1.0f / 2.5f, adat[4]);

    delete a_;
    delete cl;
}

TEST(testCLMathWrapper, perelementmult) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    float bdat[] = { 4,2.1f, 5,3,9.2f };
    CLWrapper *a_ = cl->wrap(5,adat);
    CLWrapper *b_ = cl->wrap(5,bdat);
    a_->copyToDevice();
    b_->copyToDevice();

    CLMathWrapper a(a_);
    CLMathWrapper b(b_);
    a *= b;
    a_->copyToHost();

    for(int i = 0; i < 5; i++) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR(4.0f, adat[0]);
    EXPECT_FLOAT_NEAR(6.3f, adat[1]);
    EXPECT_FLOAT_NEAR(2.5f * 9.2f, adat[4]);

//    delete a;
//    delete b;
    delete a_;
    delete b_;
    delete cl;
}

