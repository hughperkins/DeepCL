// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include "FileHelper.h"
#include "stringhelper.h"

#include "Kgsv2Loader.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

STATIC void Kgsv2Loader::getDimensions( std::string filepath, int *p_N, int *p_numPlanes, int *p_boardSize, int *p_totalImagesLinearSize ) {
    char *headerBytes = FileHelper::readBinaryChunk( filepath, 0, 1024 );
    headerBytes[1023] = 0;
    string headerString = string( headerBytes );
    vector<string> splitHeader = split( headerString, "-" );
    if( splitHeader[0] != "mlv2" ) {
        throw runtime_error( "file " + filepath + " is not an mlv2 (kgsgo) data file" );
    }
    int N = atoi( split( split( headerString, "-n=" )[1], "-" )[0] );
    int numPlanes = atoi( split( split( headerString, "-numplanes=" )[1], "-" )[0] );
    int boardSize = atoi( split( split( headerString, "-imagewidth=" )[1], "-" )[0] );
    int boardSizeRepeated = atoi( split( split( headerString, "-imageheight=" )[1], "-" )[0] );
    if( boardSize != boardSizeRepeated ) {
        throw runtime_error( "file " + filepath + " contains non-square images.  Not handled for now." );
    }
    *p_N = N;
    *p_numPlanes = numPlanes;
    *p_boardSize = boardSize;
    *p_totalImagesLinearSize = N * numPlanes * boardSize * boardSize;
}

//STATIC int Kgsv2Loader::getNumRecords( std::string filepath ) {
//    long filesize = FileHelper::getFilesize( filepath );
//    int recordsSize = filesize - 4; // because of 'END' at the end
//    int numRecords = recordsSize / getRecordSize();
//    return numRecords;
//}

//STATIC int Kgsv2Loader::loadKgs( std::string filepath, int *p_numPlanes, int *p_boardSize, unsigned char *data, int *labels ) {
//    return loadKgs( filepath, p_numPlanes, p_boardSize, data, labels, 0, getNumRecords( filepath ) );
//}

STATIC void Kgsv2Loader::load( std::string filepath, unsigned char *data, int *labels ) {
    load( filepath, data, labels, 0, 0 );
}

STATIC void Kgsv2Loader::load( std::string filepath, unsigned char *data, int *labels, int startRecord, int numRecords ) {
    int N;
    int boardSize;
    int numPlanes;
    int imagesSize;
    getDimensions( filepath, &N, &numPlanes, &boardSize, &imagesSize );
    if( numRecords == 0 ) {
        numRecords = N - startRecord;
    }
    const int boardSizeSquared = boardSize * boardSize;
    const long recordSize = getRecordSize(numPlanes, boardSize);
    long pos = (long)startRecord * recordSize + 1024 /* for header */;
    long chunkByteSize = (long)numRecords * recordSize;
//    cout << "chunkByteSize: " << chunkByteSize << endl;
    unsigned char *kgsData = reinterpret_cast<unsigned char *>( FileHelper::readBinaryChunk( filepath, pos, chunkByteSize ) );
    for( int n = 0; n < numRecords; n++ ) {
        long recordOffset = (long)n * recordSize;
//        cout << "recordOffset: " << recordOffset << endl;
        unsigned char *record = kgsData + recordOffset;
        if( record[ 0 ] != 'G' ) {
            throw std::runtime_error("alignment error, for record " + toString(n) );
        }
        if( record[ 1 ] != 'O' ) {
            throw std::runtime_error("alignment error, for record " + toString(n) );
        }
        int *p_label = reinterpret_cast< int * >( record + 2 );
//        int label = p_label[0];
        // temporary hack, until fix kgsgo dataset endianness :-)
        int label = record[5] + 256 * record[4];
//        int label = record[2+4] + 256 * record[2+3];
//        cout << "label bytes: " << (int)record[2] << " " << (int)record[3] << " " << (int)record[4] << " " << (int)record[5] << endl;
//        cout << "label: " << label << endl;
        labels[n] = label;
        unsigned char *recordImage = record + 6;
        int bitPos = 0;
        int intraRecordPos = 0;
        unsigned char thisrecordbyte = recordImage[ intraRecordPos ];
        for( int plane = 0; plane < numPlanes; plane++ ) {
            unsigned char *dataPlane = data + ( (long)n * numPlanes + plane ) * boardSizeSquared;
            for( int intraBoardPos = 0; intraBoardPos < boardSizeSquared; intraBoardPos++ ) {
                unsigned char thisbyte = ( thisrecordbyte >> ( 7 - bitPos ) ) & 1;
//                cout << "thisbyte: " << (int)thisbyte << endl;
                dataPlane[ intraBoardPos ] = thisbyte * 255;
                bitPos++;
                if( bitPos == 8 ) {
                    bitPos = 0;
                    intraRecordPos++;
                    thisrecordbyte = recordImage[ intraRecordPos ];
                }
            }
        }
    }
//    return numRecords;
}

//STATIC int Kgsv2Loader::loadKgs( std::string filepath, int *p_numPlanes, int *p_boardSize, unsigned char *data, int *labels, int recordStart, int numRecords ) {
//    long pos = (long)recordStart * getRecordSize();
//    const int recordSize = getRecordSize();
//    const int boardSize = 19;
//    const int numPlanes = 8;
//    const int boardSizeSquared = boardSize * boardSize;
//    unsigned char *kgsData = reinterpret_cast<unsigned char *>( FileHelper::readBinaryChunk( filepath, pos, (long)numRecords * recordSize ) );
//    for( int n = 0; n < numRecords; n++ ) {
//        long recordPos = n * recordSize;
//        if( kgsData[recordPos + 0 ] != 'G' ) {
//            throw std::runtime_error("alignment error, for record " + toString(n) );
//        }
//        int row = kgsData[ recordPos + 2 ];
//        int col = kgsData[ recordPos + 3 ];
//        labels[n] = row * boardSize + col;
//        for( int plane = 0; plane < numPlanes; plane++ ) {
//            for( int intraBoardPos = 0; intraBoardPos < boardSizeSquared; intraBoardPos++ ) {
//                unsigned char thisbyte = kgsData[ recordPos + intraBoardPos + 4 ];
//                thisbyte = ( thisbyte >> plane ) & 1;
//                data[ ( n * numPlanes + plane * boardSizeSquared ) + intraBoardPos ] = thisbyte;
//            }
//        }
//    }
//    *p_numPlanes = numPlanes;
//    *p_boardSize = boardSize;
//    return numRecords;
//}

STATIC int Kgsv2Loader::getRecordSize( int numPlanes, int boardSize ) {
    const int boardSizeSquared = boardSize * boardSize;
    int recordSize = 2 /* "GO" */ + 4 /* label */;
    // + boardSizeSquared;
    int numBits = numPlanes * boardSize * boardSize;
    int numBytes = ( numBits + 8 - 1 ) / 8;
    recordSize += numBytes;
    cout << "numBits " << numBits << " numBytes " << numBytes << " recordSize " << recordSize << endl;
    return recordSize;
}

