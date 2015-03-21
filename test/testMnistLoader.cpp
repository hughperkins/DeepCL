#include <iostream>

#include "gtest/gtest.h"

#include "MnistLoader.h"
#include "ImagesHelper.h"

#include "test/myasserts.h"

using namespace std;

TEST( testMnistLoader, readwriteuint ) {
    unsigned char fakedata[8];
    MnistLoader::writeUInt( fakedata, 0, 1234567890 );
    int returned = MnistLoader::readUInt( fakedata, 0 );
    assertEquals( (int)returned, 1234567890 );
}

TEST( testMnistLoader, labels ) {
    int N;
    int *labelsTrain = MnistLoader::loadLabels( "../data/mnist", "train", &N );
    EXPECT_EQ( 9, labelsTrain[4] );
    EXPECT_EQ( 5, labelsTrain[11] );
    EXPECT_EQ( 5, labelsTrain[11] );
    EXPECT_EQ( 4, labelsTrain[40013] );
    EXPECT_EQ( 4, labelsTrain[57123] );
    EXPECT_EQ( 8, labelsTrain[59999] );
    delete[] labelsTrain;

    int *labelsTest = MnistLoader::loadLabels( "../data/mnist", "t10k", &N );
    EXPECT_EQ( 4, labelsTest[4] );
    EXPECT_EQ( 4, labelsTest[1422] );
    EXPECT_EQ( 4, labelsTest[5312] );
    EXPECT_EQ( 9, labelsTest[7182] );
    EXPECT_EQ( 5, labelsTest[8327] );
    EXPECT_EQ( 6, labelsTest[9999] );
    delete[] labelsTest;
}

TEST( testMnistLoader, images ) {
    int N;
    int imageSize;
    int ***imagesTrain = MnistLoader::loadImages( "../data/mnist", "train", &N, &imageSize );
    EXPECT_EQ( 60000, N );
    EXPECT_EQ( 28, imageSize );
    int thismax = 0;
    int thismin = 0;
    for( int i = 0; i < 1000; i++ ) {
        int thisvalue = (&imagesTrain[0][0][0])[i];
        EXPECT_GT( thisvalue, -1 );
        EXPECT_LT( thisvalue, 256 );
        thismax = max( thisvalue, thismax );
        thismin = min( thisvalue, thismin );
    }
    EXPECT_EQ( 0, thismin );
    EXPECT_EQ( 255, thismax );
    ImagesHelper::deleteImages( &imagesTrain, N, imageSize );
}

