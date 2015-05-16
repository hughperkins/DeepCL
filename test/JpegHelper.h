#include <iostream>
#include <cstdio>
#include <jpeglib.h>
#include <stdexcept>
using namespace std;

class JpegHelper {
public:
    static void writeJpeg( std::string filename, int planes, int width, int height, unsigned char *values );
};

