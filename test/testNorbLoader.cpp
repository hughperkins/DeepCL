// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#ifdef PNG_AVAILABLE
#include "png++/png.hpp"
#endif //PNG_AVAILABLE

#include "loaders/NorbLoader.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

using namespace std;

TEST(SLOW_testNorbLoader, loadall) {
    int N;
    int numPlanes;
    int imageSize;
    string norbDataDir = "../data/norb";

//    string trainingFilename = "smallnorb-5x46789x9x18x6x2x96x96-training";
    string trainingFilename = "training-shuffled";
//    string trainingFilename = "testing-sampled";

    NorbLoader::getDimensions(norbDataDir + "/" + trainingFilename + "-dat.mat", &N, &numPlanes, &imageSize);
    unsigned char *images = NorbLoader::loadImages(norbDataDir + "/" + trainingFilename + "-dat.mat", &N, &numPlanes, &imageSize);
    int *labels = NorbLoader::loadLabels(norbDataDir + "/" + trainingFilename + "-cat.mat", N);
    cout << "labels here, please open testNorbLoader.png, and compare" << endl;
    for(int i = 0; i < 4; i++) {
        string thisRow = "";
        for(int j = 0; j < 4; j++) {
            thisRow += toString(labels[i*4+j]) + " ";
        }
        cout << thisRow << endl;
    }
#ifdef PNG_AVAILABLE
    png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >(imageSize * 8, imageSize * 4);
    for(int imageRow = 0; imageRow < 4; imageRow++) {
        for(int imageCol = 0; imageCol < 4; imageCol++) {
            for(int p = 0; p < 2; p++) {
                for(int i = 0; i < imageSize; i++) {
                    for(int j = 0; j < imageSize; j++) {
                           int value = images[((imageRow*4+imageCol)*2 + p) * imageSize * imageSize + i*imageSize + j];
                       (*image)[i + imageRow*imageSize][j + (imageCol*2+p)*imageSize] = png::rgb_pixel(value, value, value);
                    }
                }
            }
        }
    }
    FileHelper::remove("testNorbLoader.png");
    image->write("testNorbLoader.png");
#endif
    delete[] images;
}

TEST(DATA_testNorbLoader, load1000) {
    int N;
    int numPlanes;
    int imageSize;
    string norbDataDir = "../data/norb";

//    string trainingFilename = "smallnorb-5x46789x9x18x6x2x96x96-training";
    string trainingFilename = "training-shuffled";
//    string trainingFilename = "testing-sampled";

    unsigned char *images = NorbLoader::loadImages(norbDataDir + "/" + trainingFilename + "-dat.mat", &N, &numPlanes, &imageSize, 1000);
    cout << "N: " << N << endl;
    int *labels = NorbLoader::loadLabels(norbDataDir + "/" + trainingFilename + "-cat.mat", N);
    cout << "labels here, please open testNorbLoader.png, and compare" << endl;
    for(int i = 0; i < 4; i++) {
        string thisRow = "";
        for(int j = 0; j < 4; j++) {
            thisRow += toString(labels[i*4+j]) + " ";
        }
        cout << thisRow << endl;
    }
#ifdef PNG_AVAILABLE
    png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >(imageSize * 8, imageSize * 4);
    for(int imageRow = 0; imageRow < 4; imageRow++) {
        for(int imageCol = 0; imageCol < 4; imageCol++) {
            for(int p = 0; p < 2; p++) {
                for(int i = 0; i < imageSize; i++) {
                    for(int j = 0; j < imageSize; j++) {
                           int value = images[((imageRow*4+imageCol)*2 + p) * imageSize * imageSize + i*imageSize + j];
                       (*image)[i + imageRow*imageSize][j + (imageCol*2+p)*imageSize] = png::rgb_pixel(value, value, value);
                    }
                }
            }
        }
    }
    FileHelper::remove("testNorbLoader.png");
    image->write("testNorbLoader.png");
#endif
    delete[] images;
}

