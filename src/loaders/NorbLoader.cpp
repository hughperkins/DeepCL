// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <iostream>
#include <cstring>

#include "util/stringhelper.h"
#include "util/FileHelper.h"

#include "NorbLoader.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

STATIC void NorbLoader::getDimensions(std::string trainFilepath, int *p_N, int *p_numPlanes, int *p_imageSize) {
    char*headerBytes = FileHelper::readBinaryChunk(trainFilepath, 0, 6 * 4);
    unsigned int *headerValues = reinterpret_cast< unsigned int *>(headerBytes);

    int magic = headerValues[0];
//    std::cout << "magic: " << magic << std::endl;
    if(magic != 0x1e3d4c55) {
        throw std::runtime_error("magic value doesnt match expections: " + toString(magic));
    }
//    int ndim = headerValues[1];
    int N = headerValues[2];
    int numPlanes = headerValues[3];
    int imageSize = headerValues[4];
    int imageSizeRepeated = headerValues[5];
//    std::cout << "ndim " << ndim << " " << N << " " << numPlanes << " " << imageSize << " " << imageSizeRepeated << std::endl;
    checkSame("imageSize", imageSize, imageSizeRepeated);

//    if(maxN > 0) {
//        N = min(maxN, N);
//    }
//    int totalLinearSize = N * numPlanes * imageSize * imageSize;
    *p_N = N;
    *p_numPlanes = numPlanes;
    *p_imageSize = imageSize;
//    *p_imagesLinearSize = totalLinearSize;
}

STATIC void NorbLoader::load(std::string trainFilepath, unsigned char *images, int *labels) {
    load(trainFilepath, images, labels, 0, 0);
}

STATIC void NorbLoader::load(std::string trainFilepath, unsigned char *images, int *labels, int startN, int numExamples) {
    int N, numPlanes, imageSize;
    // I know, this could be optimized a bit, to remove the intermediate arrays...
    loadImages(images, trainFilepath, &N, &numPlanes, &imageSize, startN, numExamples);
//    int totalLinearSize = numExamples  * numPlanes * imageSize * imageSize;
//    memcpy(images, loadedImages + startN * numPlanes * imageSize * imageSize, numExamples * numPlanes * imageSize * imageSize * sizeof(unsigned char) );
    if(labels == 0) {
        return;
    }
    loadLabels(labels, replace(trainFilepath, "-dat.mat","-cat.mat"), startN, numExamples);
//    memcpy(labels, loadedLabels + startN, sizeof(int) * numExamples);
//    delete []loadedImages;
//    delete[] loadedLabels;
}

//STATIC unsigned char *NorbLoader::loadImages(std::string filepath, int *p_N, int *p_numPlanes, int *p_imageSize) {
//    return loadImages(filepath, p_N, p_numPlanes, p_imageSize, 0, 0);
//}

STATIC int *NorbLoader::loadLabels(std::string labelsfilepath, int numExamples) {
    int *labels = new int[numExamples];
    loadLabels(labels, labelsfilepath, 0, numExamples);
    return labels;
}

STATIC unsigned char *NorbLoader::loadImages(std::string filepath, int *p_N, int *p_numPlanes, int *p_imageSize) {
    return loadImages(filepath, p_N, p_numPlanes, p_imageSize, 0, 0);
}

STATIC unsigned char *NorbLoader::loadImages(std::string filepath, int *p_N, int *p_numPlanes, int *p_imageSize, int numExamples) {
    return loadImages(filepath, p_N, p_numPlanes, p_imageSize, 0, numExamples);
}

STATIC unsigned char *NorbLoader::loadImages(std::string filepath, int *p_N, int *p_numPlanes, int *p_imageSize, int startN, int numExamples) {
    getDimensions(filepath, p_N, p_numPlanes, p_imageSize);
    if(numExamples == 0) {
        numExamples = *p_N - startN;
    }
    unsigned char *images = new unsigned char[ (long)numExamples * *p_numPlanes * *p_imageSize * *p_imageSize ];
    loadImages(images, filepath, p_N, p_numPlanes, p_imageSize, startN, numExamples);
    return images;
}

// you need to allocate this yourself, before use
STATIC void NorbLoader::loadImages(unsigned char *images, std::string filepath, int *p_N, int *p_numPlanes, int *p_imageSize, int startN, int numExamples) {
    char*headerBytes = FileHelper::readBinaryChunk(filepath, 0, 6 * 4);
    unsigned int *headerValues = reinterpret_cast< unsigned int *>(headerBytes);

    int magic = headerValues[0];
//    std::cout << "magic: " << magic << std::endl;
    if(magic != 0x1e3d4c55) {
        throw std::runtime_error("magic value doesnt match expections: " + toString(magic));
    }
//    int ndim = headerValues[1];
    int N = headerValues[2];
    int numPlanes = headerValues[3];
    int imageSize = headerValues[4];
    int imageSizeRepeated = headerValues[5];
//    std::cout << "ndim " << ndim << " " << N << " " << numPlanes << " " << imageSize << " " << imageSizeRepeated << std::endl;
    checkSame("imageSize", imageSize, imageSizeRepeated);

    if(numExamples > 0 && numExamples > (N - startN) ) {
        throw runtime_error("You requested " + toString(numExamples) + " but there are only " + toString(N - startN) + " avialalbe after start N " + toString(startN) );
    }
    if(numExamples == 0) {
        numExamples = N - startN;
    }
//    if(maxN > 0) {
//        N = min(maxN, N);
//    }
    long fileStartPos = 6 * 4 + (long)startN * numPlanes * imageSize * imageSize;
    long fileReadLength = (long)numExamples * numPlanes * imageSize * imageSize;
    char *imagesAsCharArray = reinterpret_cast< char *>(images);
//    cout << "images, filestartpos " << fileStartPos << " readlength " << fileReadLength << endl;
    FileHelper::readBinaryChunk(imagesAsCharArray, filepath, fileStartPos, fileReadLength);
//    unsigned char *imagesDataUnsigned = reinterpret_cast< unsigned char *>(imagesDataSigned);

    *p_N = N;
    *p_numPlanes = numPlanes;
    *p_imageSize = imageSize;
//    return imagesDataUnsigned;
}
// you need to allocate this yourself, before use
STATIC void NorbLoader::loadLabels(int *labels, std::string filepath, int startN, int numExamples) {
    char*headerBytes = FileHelper::readBinaryChunk(filepath, 0, 6 * 5);
    unsigned int *headerValues = reinterpret_cast< unsigned int *>(headerBytes);

    int magic = headerValues[0];
//    std::cout << "magic: " << magic << std::endl;
    if(magic != 0x1e3d4c54) {
        throw std::runtime_error("magic value doesnt match expections: " + toString(magic) + " expected: " + toString(0x1e3d4c54) );
    }
//    int ndim = headerValues[1];
    int N = headerValues[2];
//    checkSame("ndim", 1, ndim);
    if(numExamples > 0 && numExamples > (N - startN) ) {
        throw runtime_error("You requested " + toString(numExamples) + " but there are only " + toString(N - startN) + " avialalbe after start N " + toString(startN) );
    }
    if(numExamples == 0) {
        numExamples = N - startN;
    }
//    N = Ntoget;

//    int totalLinearSize = N;
    char *labelsAsCharArray = reinterpret_cast< char *>(labels);
    long fileStartPos = 5 * 4 + (long)startN * 4;
    long fileReadLength = (long)numExamples * 4;
//    cout << "labels file read start " << fileStartPos << " length " << fileReadLength << endl;
    FileHelper::readBinaryChunk(labelsAsCharArray, filepath, fileStartPos, fileReadLength);
//    int *labels = reinterpret_cast< int *>(labelsAsByteArray);
//    return labels;
}
STATIC void NorbLoader::writeImages(std::string filepath, unsigned char *images, int N, int numPlanes, int imageSize) {
    int totalLinearSize = N * numPlanes * imageSize * imageSize;

    long imagesFilesize = totalLinearSize + 6 * 4; // magic, plus num dimensions, plus 4 dimensions
    char *imagesDataSigned = new char[ imagesFilesize ];
    unsigned int *imagesDataInt = reinterpret_cast< unsigned int *>(imagesDataSigned);
    unsigned char *imagesDataUnsigned = reinterpret_cast< unsigned char *>(imagesDataSigned);
    imagesDataInt[0] = 0x1e3d4c55;
    imagesDataInt[1] = 4;
    imagesDataInt[2] = N;
    imagesDataInt[3] = numPlanes;
    imagesDataInt[4] = imageSize;
    imagesDataInt[5] = imageSize;
    memcpy(imagesDataUnsigned + 6 * sizeof(int), images, totalLinearSize * sizeof(unsigned char) );
    FileHelper::writeBinary(filepath, imagesDataSigned, imagesFilesize);
}
STATIC void NorbLoader::writeLabels(std::string filepath, int *labels, int N) {
    int totalLinearSize = N;

    long imagesFilesize = totalLinearSize * 4 + 5 * 4; // magic, plus num dimensions, plus 3 dimensions
    char *imagesDataSigned = new char[ imagesFilesize ];
    unsigned int *imagesDataInt = reinterpret_cast< unsigned int *>(imagesDataSigned);
    unsigned char *imagesDataUnsigned = reinterpret_cast< unsigned char *>(imagesDataSigned);
    imagesDataInt[0] = 0x1e3d4c54;
    imagesDataInt[1] = 1;
    imagesDataInt[2] = N;
    imagesDataInt[3] = 1;
    imagesDataInt[4] = 1;
    memcpy(imagesDataUnsigned + 5 * sizeof(int), labels, totalLinearSize * sizeof(int) );
    FileHelper::writeBinary(filepath, imagesDataSigned, imagesFilesize);
}

