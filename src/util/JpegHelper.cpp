// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstdio>
extern "C" {
    #include <jpeglib.h>
}
#include <stdexcept>

#include "util/stringhelper.h"
#include "util/FileHelper.h"
#include "util/JpegHelper.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLIC STATIC void JpegHelper::write(std::string filename, int planes, int width, int height, unsigned char *values) {
    unsigned char *image_buffer = new unsigned char[width * height * planes];
//    for(int i = 0 ; i < 28 *28 *3; i++) {
//=        image_buffer[i] = i * 255 / 28 * 28 / 3;
        //image_buffer[i] = 128;
//    }
    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            for(int plane = 0; plane < planes; plane++) {
                image_buffer[row*width*planes + col*planes + plane] = values[plane*width*height + row*width + col];
//                if(( y % 2 == 0 & x % 2 == 0) ) {
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
    if ((outfile = fopen(FileHelper::localizePath(filename).c_str(), "wb")) == NULL) {
        throw runtime_error("can't open "  + filename);
    }
    jpeg_stdio_dest(&cinfo, outfile);


    cinfo.image_width = width;      /* image width and height, in pixels */
    cinfo.image_height = height;
    cinfo.input_components = planes;     /* # of color components per pixel */
    if(planes == 3) {
        cinfo.in_color_space = JCS_RGB; /* colorspace of input image */
    } else if(planes == 1) {
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

PUBLIC STATIC void JpegHelper::read(std::string filename, int planes, int width, int height, unsigned char *values) {
    unsigned char *image_buffer = new unsigned char[width * height * planes];

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

//    string filename = "foo.jpeg";
    FILE * infile;
    if ((infile = fopen(FileHelper::localizePath(filename).c_str(), "rb")) == NULL) {
        throw runtime_error("can't open "  + filename);
    }
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    jpeg_start_decompress(&cinfo);
    if((int)cinfo.output_width != width) {
        throw runtime_error("error reading " + filename + ":" + 
            " width is " + toString(cinfo.output_width) + 
            " and not " + toString(width) );
    }
    if((int)cinfo.output_height != height) {
        throw runtime_error("error reading " + filename + ":" + 
            " height is " + toString(cinfo.output_height) + 
            " and not " + toString(height) );
    }
    if((int)cinfo.output_components != planes) {
        throw runtime_error("error reading " + filename + ":" + 
            " planes is " + toString(cinfo.output_components) + 
            " and not " + toString(planes) );
    }

    JSAMPROW row_pointer[1];        /* pointer to a single row */
    int row_stride;                 /* physical row width in buffer */

    row_stride = cinfo.image_width * planes;   /* JSAMPLEs per row in image_buffer */

    while (cinfo.output_scanline < cinfo.output_height) {
        row_pointer[0] = & image_buffer[cinfo.output_scanline * row_stride];
        jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    fclose(infile);

    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            for(int plane = 0; plane < planes; plane++) {
                values[plane*width*height + row*width + col] = image_buffer[row*width*planes + col*planes + plane];
            }
        }
    }

    delete[] image_buffer;
}

