#include <iostream>

#include "gtest/gtest.h"

#include "ImagesHelper.h"
#include "MnistLoader.h"
#include "ImagePng.h"

using namespace std;

//int main( int argc, char *argv[] ) {
//    MemoryChecker memoryChecker;
TEST( testimageshelper, imageshelper ) {
    int ***images = ImagesHelper::allocateImages( 5, 11 );
    ImagesHelper::deleteImages( &images, 5, 11 );

//    int N = 1000;
    int N;
    int imageSize;
    images = MnistLoader::loadImages( "../data/mnist", "train", &N, &imageSize );
    int ***results = ImagesHelper::allocateImages( N, imageSize );

    int *images_1d = &(images[0][0][0]);
    for( int i = 0; i < N * imageSize * imageSize; i++ ) {
        images_1d[i] = 0x55;
    }

    ImagePng::writeImagesToPng( "foo.png", images, min(100, N ), imageSize );
    ImagePng::writeImagesToPng( "foo.png", images, min(100, N ), imageSize );
    ImagePng::writeImagesToPng( "foo.png", images, min(100, N ), imageSize );
    ImagePng::writeImagesToPng( "foo.png", images, min(100, N ), imageSize );

    ImagesHelper::deleteImages( &results, N, imageSize );
    ImagesHelper::deleteImages( &images, N, imageSize );
//    return 0;
}

