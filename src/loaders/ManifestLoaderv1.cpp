// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "util/FileHelper.h"
#include "util/stringhelper.h"
#include "ManifestLoaderv1.h"
#include "util/JpegHelper.h"

#include "DeepCLDllExport.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLIC STATIC bool ManifestLoaderv1::isFormatFor( std::string imagesFilepath ) {
    cout << "ManifestLoaderv1 checking format for " << imagesFilepath << endl;
    char *headerBytes = FileHelper::readBinaryChunk( imagesFilepath, 0, 1024 );
    string sigString = "# format=deepcl-jpeg-list-v1 ";
    headerBytes[sigString.length()] = 0;
    bool matched = string( headerBytes ) == sigString;
    cout << "matched: " << matched << endl;
    return matched;
}
PUBLIC ManifestLoaderv1::ManifestLoaderv1( std::string imagesFilepath ) {
    init( imagesFilepath, true );    
}
PUBLIC ManifestLoaderv1::ManifestLoaderv1( std::string imagesFilepath, bool includeLabels ) {
    init( imagesFilepath, includeLabels );
}
PRIVATE void ManifestLoaderv1::init( std::string imagesFilepath, bool includeLabels ) {
    this->includeLabels = includeLabels;
    this->imagesFilepath = imagesFilepath;
    // by reading the number of lines in the manifest, we can get the number of examples, *p_N
    // number of planes is .... 1
    // imageSize is ...

    if( !isFormatFor( imagesFilepath ) ) {
        throw runtime_error( "file " + imagesFilepath + " is not a deepcl-jpeg-list-v1 manifest file" );
    }

    ifstream infile( imagesFilepath );
    char lineChars[1024];
    infile.getline( lineChars, 1024 ); // skip first, header, line
    string firstLine = string( lineChars );
//    cout << "firstline: [" << firstLine << "]" << endl;
    vector<string> splitLine = split( firstLine, " " );
    N = readIntValue( splitLine, "N" );
    planes = readIntValue( splitLine, "planes" );
    size = readIntValue( splitLine, "width" );
    int imageSizeRepeated = readIntValue( splitLine, "height" );
    if( size != imageSizeRepeated ) {
        throw runtime_error( "file " + imagesFilepath + " contains non-square images.  Not handled for now." );
    }
    // now we should load into memory, since the file is not fixed-size records, and cannot be loaded partially easily

    files = new string[N];
    labels = new int[N];

    int n = 0;
    while( infile ) {
        infile.getline( lineChars, 1024 );
        if( !infile ) {
            break;
        }
        string line = string( lineChars );
        if( line == "" ) {
            continue;
        }
        vector<string> splitLine = split(line, " ");
        if( (int)splitLine.size() == 0 ) {
            continue;
        }
        if( includeLabels && (int)splitLine.size() != 2 ) { 
            throw runtime_error("Error reading " + imagesFilepath + ".  Following line not parseable:\n" + line );
        }
        string jpegFile = splitLine[0];
        files[n] = jpegFile;
        if( includeLabels ) {
            int label = atoi(splitLine[1]);
        labels[n] = label;
        }
//        cout << "file " << jpegFile << " label=" << label << endl;
        n++;
    }
    infile.close();

    cout << "manifest " << imagesFilepath << " read. N=" << N << " planes=" << planes << " size=" << size << endl;
}
PUBLIC VIRTUAL std::string ManifestLoaderv1::getType() {
    return "ManifestLoaderv1";
}
PUBLIC VIRTUAL int ManifestLoaderv1::getImageCubeSize() {
    return planes * size * size;
}
PUBLIC VIRTUAL int ManifestLoaderv1::getN() {
    return N;
}
PUBLIC VIRTUAL int ManifestLoaderv1::getPlanes() {
    return planes;
}
PUBLIC VIRTUAL int ManifestLoaderv1::getImageSize() {
    return size;
}
int ManifestLoaderv1::readIntValue( std::vector< std::string > splitLine, std::string key ) {
    for( int i = 0; i < (int)splitLine.size(); i++ ) {
        vector<string> splitPair = split( splitLine[i], "=" );
        if( (int)splitPair.size() == 2 ) {
            if( splitPair[0] == key ) {
                return atoi( splitPair[1] );
            }
        }
    }
    throw runtime_error("Key " + key + " not found in file header" );
}
PUBLIC VIRTUAL void ManifestLoaderv1::load( unsigned char *data, int *labels, int startRecord, int numRecords ) {
    int imageCubeSize = planes * size * size;
//    cout << "ManifestLoaderv1, loading " << numRecords << " jpegs" << endl;
    for( int localN = 0; localN < numRecords; localN++ ) {
        int globalN = localN + startRecord;
        JpegHelper::read( files[globalN], planes, size, size, data + localN * imageCubeSize );
        if( labels != 0 ) {
            if( !includeLabels ) {
                throw runtime_error( "ManifestLoaderv1: labels reqested in load() method, but not activated in constructor" );
            }
            labels[localN] = this->labels[globalN];
        }
    }
}

