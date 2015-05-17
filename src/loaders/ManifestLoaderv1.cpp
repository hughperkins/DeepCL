// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <string>
#include <cstring>
#include <stdexcept>

#include "util/FileHelper.h"
#include "util/stringhelper.h"
#include "ManifestLoaderv1.h"

#include "DeepCLDllExport.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLIC STATIC bool ManifestLoaderv1::isFormatFor( std::string imagesFilepath ) {
    char *headerBytes = FileHelper::readBinaryChunk( imagesFilepath, 0, 1024 );
    headerBytes[1024] = 0;
    char const*header = "# format=deepcl-jpeg-list-v1 ";
    return string( headerBytes ) == string( header );
}
PUBLIC ManifestLoaderv1::ManifestLoaderv1( std::string imagesFilepath ) {
    this->imagesFilepath = imagesFilepath;
    // by reading the number of lines in the manifest, we can get the number of examples, *p_N
    // number of planes is .... 1
    // imageSize is ...

    char *headerBytes = FileHelper::readBinaryChunk( imagesFilepath, 0, 1024 );
    headerBytes[1023] = 0;
    string headerString = string( headerBytes );
    vector<string> splitHeader = split( headerString, " " );
    if( splitHeader[0] != "#" || splitHeader[1] != "format=deepcl-jpeg-list-v1" ) {
        throw runtime_error( "file " + imagesFilepath + " is not a deepcl-jpeg-list-v1 manifest file" );
    }
    string firstLine = split( headerBytes, "\n" )[0];
    cout << "firstline: [" << firstLine << "]" << endl;
    vector<string> splitLine = split( firstLine, " " );
    N = readIntValue( splitLine, "N" );
    planes = readIntValue( splitLine, "numplanes" );
    size = readIntValue( splitLine, "width" );
    int imageSizeRepeated = readIntValue( splitLine, "height" );
    if( size != imageSizeRepeated ) {
        throw runtime_error( "file " + imagesFilepath + " contains non-square images.  Not handled for now." );
    }
}
PUBLIC VIRTUAL std::string ManifestLoaderv1::getType() {
    return "ManifestLoaderv1";
}
PUBLIC VIRTUAL int ManifestLoaderv1::getImageCubeSize() {
    return planes * size * size;
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
VIRTUAL void ManifestLoaderv1::getDimensions() {
}

PUBLIC VIRTUAL void ManifestLoaderv1::load( unsigned char *data, int *labels, int startRecord, int numRecords ) {
    // we're going to have to read the names of the files from a file which doesnt have fixed-length fields
    // probably we should cache this somehow, or perhaps write to a file that does have fixed-length fields

}

