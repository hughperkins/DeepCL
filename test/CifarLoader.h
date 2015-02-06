// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class CifarLoader {
public:
    static int getNumExamples( std::string filepath );
    static int getImagesSize( std::string filepath ); // you can use this to help allocate the images array
    static void load( std::string filepath, unsigned char *images, int *labels ); // you need to pre-allocate these arrays
};

