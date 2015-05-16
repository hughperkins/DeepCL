#include <iostream>
#include <cstdio>
#include <jpeglib.h>
#include <stdexcept>
using namespace std;

#include "util/stringhelper.h"
#include "test/JpegHelper.h"

void JpegHelper::writeJpeg( string filename, int planes, int width, int height, unsigned char *values ) {
    unsigned char *image_buffer = new unsigned char[width * height * planes];
//    for( int i = 0 ; i < 28 *28 *3; i++ ) {
//=        image_buffer[i] = i * 255 / 28 * 28 / 3;
        //image_buffer[i] = 128;
//    }
    for( int row = 0; row < height; row++ ) {
        for( int col = 0; col < width; col++ ) {
            for( int plane = 0; plane < planes; plane++ ) {
                image_buffer[row*width*planes + col*planes + plane] = values[plane*width*height + row*width + col];
//                if( ( y % 2 == 0 & x % 2 == 0 ) ) {
//                    image_buffer[x * 28 * 3 + y *3 + c] = 255;
//                }
            }
        }
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

//    string filename = "foo.jpeg";
    FILE * outfile;
    if ((outfile = fopen(filename.c_str(), "wb")) == NULL) {
        throw runtime_error( "can't open "  + filename );
    }
    jpeg_stdio_dest(&cinfo, outfile);


    cinfo.image_width = width;      /* image width and height, in pixels */
    cinfo.image_height = height;
    cinfo.input_components = planes;     /* # of color components per pixel */
    if( planes == 3 ) {
        cinfo.in_color_space = JCS_RGB; /* colorspace of input image */
    } else if( planes == 1 ) {
        cinfo.in_color_space = JCS_GRAYSCALE;
    } else {
        throw runtime_error("num planes " + toString(planes) + " not handled");
    }

    jpeg_set_defaults(&cinfo);

    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];        /* pointer to a single row */
    int row_stride;                 /* physical row width in buffer */

    row_stride = cinfo.image_width * planes;   /* JSAMPLEs per row in image_buffer */

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = & image_buffer[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    fclose(outfile);
    delete[] image_buffer;
}

