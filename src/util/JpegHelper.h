#include <iostream>
#include <cstdio>
#include <jpeglib.h>
#include <stdexcept>
using namespace std;

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class JpegHelper {

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    STATIC void write( string filename, int planes, int width, int height, unsigned char *values );
    STATIC void read( string filename, int planes, int width, int height, unsigned char *values );

    // [[[end]]]
};

