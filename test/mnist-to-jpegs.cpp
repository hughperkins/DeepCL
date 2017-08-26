// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

// used to test loading jpegs from imagenet etc
// this will take mnist dataset, and generate jpegs, in directories, one
// directory per category, and an appropriate manifest file
// when we read the images, we will assume all images have the same
// dimensions

// note that the labels are part of the manifest, which will have the following
// format:
// # format=deepcl-jpeg-list-v1 planes=1 width=28 height=28 N=10000
// [relative file path]  [category number, zero-based]

#include <iostream>
#include <fstream>

#include "loaders/GenericLoader.h"
#include "util/FileHelper.h"
#include "util/mt19937defs.h"
#include "util/stringhelper.h"
#include "util/JpegHelper.h"

using namespace std;

void run(string mnistImagesFile, string outDirectory, int numExamples) {
    int N, planes, size;
    GenericLoader::getDimensions( mnistImagesFile.c_str(), &N, &planes, &size );
    float *imageData = new float[ N * planes * size * size ];
    int *labels = new int[N];
    GenericLoader::load( mnistImagesFile.c_str(), imageData, labels, 0, numExamples );

    // now we've loaded the data, write it out in deepcl-jpeg-list-v1 format
    // we need to do the following:
    // - create a sub-folder for each category
    // - write each image as a jpeg to this folder
    // - write the manifest

    // creating the folders should be easy enough... I guess... ?
    MT19937 myrandom;
    int inputCubeSize = planes * size * size;
    uchar *ucharValues = new uchar[ inputCubeSize ];
    for( int i = 0; i < 10; i++ ) {
        myrandom.seed((unsigned int)(i + 1));
        int thisref = myrandom() % 10000;
        string folderPath = outDirectory + "/R131" + toString( thisref ); // make the name a bit imagenet-like
        if( !FileHelper::folderExists( folderPath ) ) {
            FileHelper::createDirectory( folderPath );
        }
    }
    // write the jpegs
    ofstream manifest(outDirectory + "/manifest.txt");
    manifest << "# format=deepcl-jpeg-list-v1 N=" << numExamples << " planes=" << planes << " width=" << size << " height=" << size << endl;
    for( int n = 0; n < numExamples; n++ ) {
        int label = labels[n];
        myrandom.seed((unsigned int)(label + 1));
        int thisref = myrandom() % 10000;
        string folderPath = outDirectory + "/R131" + toString( thisref ); // make the name a bit imagenet-like
        float *inputCube = imageData + n * inputCubeSize;
        for( int j = 0; j < inputCubeSize; j++ ) {
            ucharValues[j] = inputCube[j];
        }
        string filePath = folderPath + "/" + toString( n ) + ".JPEG";
        JpegHelper::write( filePath, planes, size, size, ucharValues );
        manifest << filePath << " " << label << endl;
    }
    manifest.close();

    delete[] ucharValues;
    delete[] labels;
    delete[] imageData;
}

int main( int argc, char *argv[] ) {
    if( argc != 4 ) {
        cout << "Usage: " << argv[0] << " [mnist images file (input)] [output directory] [num examples]" << endl;
        return 1;
    }
    string mnistImagesFile = argv[1];
    string outDirectory = argv[2];
    int numExamples = atoi(argv[3]);
    try {
        run(mnistImagesFile, outDirectory, numExamples);
    } catch(runtime_error e) {
        cout << "something went wrong: " << e.what() << endl;
        return -1;
    }

    return 0;
}

