// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <random>

#include "util/stringhelper.h"
#include "MnistLoader.h"

using namespace std;

#undef STATIC
#define STATIC

STATIC void MnistLoader::getDimensions(std::string imagesFilePath, 
    int *p_numExamples, int *p_numPlanes, int *p_imageSize) {
    char*headerBytes = FileHelper::readBinaryChunk(imagesFilePath, 0, 4 * 4);
    unsigned char *headerValues = reinterpret_cast< unsigned char *>(headerBytes);
    *p_numExamples = readUInt(headerValues, 1);
    *p_numPlanes = 1;
    *p_imageSize = readUInt(headerValues, 2);
    int imageSizeRepeat = readUInt(headerValues, 3);
    if(*p_imageSize != imageSizeRepeat) {
        throw runtime_error("error reading mnist-format file " + imagesFilePath + ": height and width not equal.  We only support square images currently.");
    }

    delete[] headerBytes;
}
// images and labels should already have been allocated, so we just need to read the data
// in
// oh, except the data is stored as unsigned char, not int, so need to convert
// new: if labels is 0, then it wont read labels
STATIC void MnistLoader::load(std::string imagesFilePath, unsigned char *images, int *labels, int startN, int numExamples) {
    int N, numPlanes, imageSize;
    getDimensions(imagesFilePath, &N, &numPlanes, &imageSize);
    if(numExamples == 0) {
        numExamples = N - startN;
    }
    long fileStartPos = 4 * 4 + (long)startN * numPlanes * imageSize * imageSize;
    long fileReadLength = (long)numExamples * numPlanes * imageSize * imageSize;
    char *imagesAsCharArray = reinterpret_cast< char *>(images);
    FileHelper::readBinaryChunk(imagesAsCharArray, imagesFilePath, fileStartPos, fileReadLength);

    // now do labels...
    if(labels == 0) {
        return;
    }
    string labelsFilePath = replace(imagesFilePath, "-images-idx3-ubyte", "-labels-idx1-ubyte");
    labelsFilePath = replace(labelsFilePath, "-images.idx3-ubyte", "-labels.idx1-ubyte");
//    cout << "labelsfilepath: " << labelsFilePath << endl;
    
    fileStartPos = 2 * 4 + (long)startN;
    fileReadLength = (long)numExamples;
    char *labelsAsCharArray = new char[fileReadLength];
    unsigned char *labelsAsUCharArray = reinterpret_cast< unsigned char *>(labelsAsCharArray);
//    cout << "labels path " << labelsFilePath << " startpos " << fileStartPos << " read length " << fileReadLength << endl;
    FileHelper::readBinaryChunk(labelsAsCharArray, labelsFilePath, fileStartPos, fileReadLength);
    for(int i = 0; i < numExamples; i++) {
        labels[i] = labelsAsUCharArray[i];
    }
    delete[]labelsAsCharArray;
}
//STATIC int **MnistLoader::loadImage(std::string dir, std::string set, int idx, int *p_size) {
//    long imagesFilesize = 0;
//    long labelsFilesize = 0;
//    char *imagesDataSigned = FileHelper::readBinary(dir + "/" + set + "-images-idx3-ubyte", &imagesFilesize);
//    char *labelsDataSigned = FileHelper::readBinary(dir + "/" + set + "-labels-idx1-ubyte", &labelsFilesize);
//    unsigned char *imagesData = reinterpret_cast< unsigned char *>(imagesDataSigned);
////        unsigned char *labelsData = reinterpret_cast< unsigned char *>(labelsDataSigned);

////    int numImages = readUInt(imagesData, 1);
//    int numRows = readUInt(imagesData, 2);
//    int numCols = readUInt(imagesData, 3);
//    *p_size = numRows;
////    std::cout << "numimages " << numImages << " " << numRows << "*" << numCols << std::endl;

//    int **image = ImageHelper::allocateImage(numRows);
//    for(int i = 0; i < numRows; i++) {
//        for(int j = 0; j < numRows; j++) {
//            image[i][j] = (int)imagesData[idx * numRows * numCols + i * numCols + j];
//        }
//    }
//    delete[] imagesDataSigned;
//    delete[] labelsDataSigned;
//    return image;
//}
//STATIC int ***MnistLoader::loadImages(std::string dir, std::string set, int *p_numImages, int *p_size) {
//    long imagesFilesize = 0;
//    char *imagesDataSigned = FileHelper::readBinary(dir + "/" + set + "-images-idx3-ubyte", &imagesFilesize);
//    unsigned char *imagesData = reinterpret_cast<unsigned char *>(imagesDataSigned);
//    int totalNumImages = readUInt(imagesData, 1);
//    int numRows = readUInt(imagesData, 2);
//    int numCols = readUInt(imagesData, 3);
////        *p_numImages = min(100,totalNumImages);
//    *p_numImages = totalNumImages;
//    *p_size = numRows;
////    std::cout << "totalNumImages " << *p_numImages << " " << *p_size << "*" << numCols << std::endl;
//    int ***images = ImagesHelper::allocateImages(*p_numImages, numRows);
//    for(int n = 0; n < *p_numImages; n++) {
//        for(int i = 0; i < numRows; i++) {
//            for(int j = 0; j < numRows; j++) {
//                images[n][i][j] = (int)imagesData[16 + n * numRows * numCols + i * numCols + j];
//            }
//        }
//    }
//    delete[] imagesDataSigned;
//    return images;
//}
STATIC int *MnistLoader::loadLabels(std::string dir, std::string set, int *p_numImages) {
    long labelsFilesize = 0;
    char *labelsDataSigned = FileHelper::readBinary(dir + "/" + set + "-labels-idx1-ubyte", &labelsFilesize);
    unsigned char *labelsData = reinterpret_cast<unsigned char *>(labelsDataSigned);
    int totalNumImages = readUInt(labelsData, 1);
  //  *p_numImages = min(100,totalNumImages);
    *p_numImages = totalNumImages;
//    std::cout << "set " << set << " num labels " << *p_numImages << std::endl;
    int *labels = new int[*p_numImages];
    for(int n = 0; n < *p_numImages; n++) {
       labels[n] = (int)labelsData[8 + n];
    }
    delete[] labelsDataSigned;
    return labels;
}

STATIC int MnistLoader::readUInt(unsigned char *data, int location) {
    unsigned int value = 0;
    for(int i = 0; i < 4; i++) {
        int thisbyte = data[location*4+i];
        value += thisbyte << ((3-i) * 8);
    }
//    std::cout << "readUint[" << location << "]=" << value << std::endl;
    return value;
}

STATIC void MnistLoader::writeUInt(unsigned char *data, int location, int value) {
    for(int i = 0; i < 4; i++) {
        data[location*4+i] = ((value >> ((3-i)*8))&255);
    }
}

