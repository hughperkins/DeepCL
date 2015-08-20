// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstdio>
#include <string>

#define VIRTUAL virtual
#define STATIC static

#include "DeepCLDllExport.h"

// give it filepaths with '/', and it will replace them with \\, if WIN32 is defined (ie, on Windows)
class DeepCL_EXPORT FileHelper {

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    STATIC char *readBinary(std::string filepath, long *p_filesize);
    STATIC long getFilesize(std::string filepath);
    STATIC char *readBinaryChunk(std::string filepath, long start, long length);
    STATIC void readBinaryChunk(char *targetArray, std::string filepath, long start, long length);
    STATIC void writeBinary(std::string filepath, char const*data, long filesize);
    STATIC void writeBinaryChunk(std::string filepath, char const*data, long startPos, long filesize);
    STATIC bool exists(const std::string filepath);
    STATIC void rename(std::string oldname, std::string newname);
    STATIC void remove(std::string filename);
    STATIC std::string localizePath(std::string path);
    STATIC std::string pathSeparator();
    STATIC void createDirectory(std::string path);
    STATIC bool folderExists(std::string path);

    // [[[end]]]
};

