// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#define VIRTUAL virtual
#define STATIC static

class KgsLoader {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC int getNumRecords( std::string filepath );
    STATIC int loadKgs( std::string filepath, int *p_numPlanes, int *p_imageSize, unsigned char *data, int *labels );
    STATIC int loadKgs( std::string filepath, int *p_numPlanes, int *p_imageSize, unsigned char *data, int *labels, int recordStart, int numRecords );
    STATIC int getRecordSize();

    // [[[end]]]
};

