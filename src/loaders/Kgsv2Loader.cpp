// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include "util/FileHelper.h"
#include "util/stringhelper.h"

#include "Kgsv2Loader.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

STATIC void Kgsv2Loader::getDimensions(std::string filepath, int *p_N, int *p_numPlanes, int *p_imageSize) {
    char *headerBytes = FileHelper::readBinaryChunk(filepath, 0, 1024);
    headerBytes[1023] = 0;
    string headerString = string(headerBytes);
    vector<string> splitHeader = split(headerString, "-");
    if(splitHeader[0] != "mlv2") {
        throw runtime_error("file " + filepath + " is not an mlv2 (kgsgo) data file");
    }
    int N = atoi(split(split(headerString, "-n=")[1], "-")[0]);
    int numPlanes = atoi(split(split(headerString, "-numplanes=")[1], "-")[0]);
    int imageSize = atoi(split(split(headerString, "-imagewidth=")[1], "-")[0]);
    int imageSizeRepeated = atoi(split(split(headerString, "-imageheight=")[1], "-")[0]);
    if(imageSize != imageSizeRepeated) {
        throw runtime_error("file " + filepath + " contains non-square images.  Not handled for now.");
    }
    *p_N = N;
    *p_numPlanes = numPlanes;
    *p_imageSize = imageSize;
//    *p_totalImagesLinearSize = N * numPlanes * imageSize * imageSize;
}

//STATIC int Kgsv2Loader::getNumRecords(std::string filepath) {
//    long filesize = FileHelper::getFilesize(filepath);
//    int recordsSize = filesize - 4; // because of 'END' at the end
//    int numRecords = recordsSize / getRecordSize();
//    return numRecords;
//}

//STATIC int Kgsv2Loader::loadKgs(std::string filepath, int *p_numPlanes, int *p_imageSize, unsigned char *data, int *labels) {
//    return loadKgs(filepath, p_numPlanes, p_imageSize, data, labels, 0, getNumRecords(filepath) );
//}

STATIC void Kgsv2Loader::load(std::string filepath, unsigned char *data, int *labels) {
    load(filepath, data, labels, 0, 0);
}

STATIC void Kgsv2Loader::load(std::string filepath, unsigned char *data, int *labels, int startRecord, int numRecords) {
    int N;
    int imageSize;
    int numPlanes;
//    int imagesSize;
    getDimensions(filepath, &N, &numPlanes, &imageSize);
    if(numRecords == 0) {
        numRecords = N - startRecord;
    }
    const int imageSizeSquared = imageSize * imageSize;
    const long recordSize = getRecordSize(numPlanes, imageSize);
    long pos = (long)startRecord * recordSize + 1024 /* for header */;
    long chunkByteSize = (long)numRecords * recordSize;
//    cout << "chunkByteSize: " << chunkByteSize << endl;
    unsigned char *kgsData = reinterpret_cast<unsigned char *>(FileHelper::readBinaryChunk(filepath, pos, chunkByteSize) );
    for(int n = 0; n < numRecords; n++) {
        long recordOffset = (long)n * recordSize;
//        cout << "recordOffset: " << recordOffset << endl;
        unsigned char *record = kgsData + recordOffset;
        if(record[ 0 ] != 'G') {
            throw std::runtime_error("alignment error, for record " + toString(n));
        }
        if(record[ 1 ] != 'O') {
            throw std::runtime_error("alignment error, for record " + toString(n));
        }
        if(labels != 0) {
            int *p_label = reinterpret_cast< int * >(record + 2);
            int label = p_label[0];
            labels[n] = label;
            if(label < 0) {
                throw runtime_error("Error: label " + toString(labels) + " is negative");
            }
        }
        unsigned char *recordImage = record + 6;
        int bitPos = 0;
        int intraRecordPos = 0;
        unsigned char thisrecordbyte = recordImage[ intraRecordPos ];
        for(int plane = 0; plane < numPlanes; plane++) {
            unsigned char *dataPlane = data + ((long)n * numPlanes + plane) * imageSizeSquared;
            for(int intraImagePos = 0; intraImagePos < imageSizeSquared; intraImagePos++) {
                unsigned char thisbyte = (thisrecordbyte >> (7 - bitPos) ) & 1;
//                cout << "thisbyte: " << (int)thisbyte << endl;
                dataPlane[ intraImagePos ] = thisbyte * 255;
                bitPos++;
                if(bitPos == 8) {
                    bitPos = 0;
                    intraRecordPos++;
                    thisrecordbyte = recordImage[ intraRecordPos ];
                }
            }
        }
    }
    delete[] kgsData;
//    return numRecords;
}

//STATIC int Kgsv2Loader::loadKgs(std::string filepath, int *p_numPlanes, int *p_imageSize, unsigned char *data, int *labels, int recordStart, int numRecords) {
//    long pos = (long)recordStart * getRecordSize();
//    const int recordSize = getRecordSize();
//    const int imageSize = 19;
//    const int numPlanes = 8;
//    const int imageSizeSquared = imageSize * imageSize;
//    unsigned char *kgsData = reinterpret_cast<unsigned char *>(FileHelper::readBinaryChunk(filepath, pos, (long)numRecords * recordSize) );
//    for(int n = 0; n < numRecords; n++) {
//        long recordPos = n * recordSize;
//        if(kgsData[recordPos + 0 ] != 'G') {
//            throw std::runtime_error("alignment error, for record " + toString(n));
//        }
//        int row = kgsData[ recordPos + 2 ];
//        int col = kgsData[ recordPos + 3 ];
//        labels[n] = row * imageSize + col;
//        for(int plane = 0; plane < numPlanes; plane++) {
//            for(int intraImagePos = 0; intraImagePos < imageSizeSquared; intraImagePos++) {
//                unsigned char thisbyte = kgsData[ recordPos + intraImagePos + 4 ];
//                thisbyte = (thisbyte >> plane) & 1;
//                data[ (n * numPlanes + plane * imageSizeSquared) + intraImagePos ] = thisbyte;
//            }
//        }
//    }
//    *p_numPlanes = numPlanes;
//    *p_imageSize = imageSize;
//    return numRecords;
//}

STATIC int Kgsv2Loader::getRecordSize(int numPlanes, int imageSize) {
//    const int imageSizeSquared = imageSize * imageSize;
    int recordSize = 2 /* "GO" */ + 4 /* label */;
    // + imageSizeSquared;
    int numBits = numPlanes * imageSize * imageSize;
    int numBytes = (numBits + 8 - 1) / 8;
    recordSize += numBytes;
//    cout << "numBits " << numBits << " numBytes " << numBytes << " recordSize " << recordSize << endl;
    return recordSize;
}

