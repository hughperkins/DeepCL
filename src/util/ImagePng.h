// Copyright Hugh Perkins 2013 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once
#include <cmath>

#include "png++/png.hpp"

class ImagePng {
public:
    static int getImageMax(int ** image, int imageSize) {
        int maxvalue = 0;
        for(int i = 0; i < imageSize; i++) {
            for(int j = 0; j < imageSize; j++) {
                maxvalue = std::max(maxvalue, image[i][j]);
            }
        }
        return maxvalue;
    }

    static float getImageMax(float ** image, int imageSize) {
        float maxvalue = 0;
        for(int i = 0; i < imageSize; i++) {
            for(int j = 0; j < imageSize; j++) {
                maxvalue = std::max(maxvalue, image[i][j]);
            }
        }
        return maxvalue;
    }

    static float getImageMax(float const* image, int imageSize) {
        float maxvalue = 0;
        for(int i = 0; i < imageSize; i++) {
            for(int j = 0; j < imageSize; j++) {
                maxvalue = std::max(maxvalue, image[i*imageSize + j]);
            }
        }
        return maxvalue;
    }
    static float getImageMin(float const* image, int imageSize) {
        float minvalue = 0;
        for(int i = 0; i < imageSize; i++) {
            for(int j = 0; j < imageSize; j++) {
                minvalue = std::min(minvalue, image[i*imageSize + j]);
            }
        }
        return minvalue;
    }
//    static float getImageMax(unsigned char const* image, int imageSize) {
//        float maxvalue = 0;
//        for(int i = 0; i < imageSize; i++) {
//            for(int j = 0; j < imageSize; j++) {
//                maxvalue = std::max(maxvalue, image[i*imageSize + j]);
//            }
//        }
//        return maxvalue;
//    }
//    static float getImageMin(unsigned char const* image, int imageSize) {
//        float minvalue = 0;
//        for(int i = 0; i < imageSize; i++) {
//            for(int j = 0; j < imageSize; j++) {
//                minvalue = std::min(minvalue, image[i*imageSize + j]);
//            }
//        }
//        return minvalue;
//    }

    static void writeImageToPng(std::string filename, int **inimage, int imageSize) {
        int maxvalue = getImageMax(inimage, imageSize);
        png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >(imageSize, imageSize);
        for(int i = 0; i < imageSize; i++) {
            for(int j = 0; j < imageSize; j++) {
               (*image)[i][j] = png::rgb_pixel(inimage[i][j] * 255 / maxvalue, inimage[i][j] * 255 / maxvalue, inimage[i][j] * 255 / maxvalue);
            }
        }
        remove(filename.c_str());
        image->write(filename);
        delete image;
    }

    static void writeImagesToPng(std::string filename, int ***images, int numImages, int imageSize) {
        int cols = sqrt(numImages);
        if(cols * cols < numImages) {
            cols++;
        }
        int rows = (numImages + cols - 1) / cols;
        std::cout << "numImages " << numImages << " rows " << rows << " cols " << cols << std::endl;
        png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >(imageSize * rows, imageSize * cols);


        for(int x = 0; x < cols; x++) {
           for(int y = 0; y < rows; y++) {
                if(x * rows + y >= numImages) {
                    continue;
                }
//                cout << "image at x " << x << " y " << y << endl;
                int **imagearray = images[x*rows + y];
                int maxvalue = std::max(1, getImageMax(imagearray, imageSize) );
                for(int i = 0; i < imageSize; i++) {
                    for(int j = 0; j < imageSize; j++) {
                       (*image)[x*imageSize + i][y*imageSize + j] = png::rgb_pixel(imagearray[i][j] * 255 / maxvalue, imagearray[i][j] * 255 / maxvalue, imagearray[i][j] * 255 / maxvalue);
                    }
                }

            }
         }
        remove(filename.c_str());
        image->write(filename);
        delete image;
    }

    static void writeImagesToPng(std::string filename, float ***images, int numImages, int imageSize) {
        int cols = sqrt(numImages);
        if(cols * cols < numImages) {
            cols++;
        }
        int rows = (numImages + cols - 1) / cols;
        std::cout << "numImages " << numImages << " rows " << rows << " cols " << cols << std::endl;
        png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >(imageSize * rows, imageSize * cols);


        for(int x = 0; x < cols; x++) {
           for(int y = 0; y < rows; y++) {
                if(x * rows + y >= numImages) {
                    continue;
                }
//                cout << "image at x " << x << " y " << y << endl;
                float **imagearray = images[x*rows + y];
                float maxvalue = std::max(1.0f, getImageMax(imagearray, imageSize) );
                for(int i = 0; i < imageSize; i++) {
                    for(int j = 0; j < imageSize; j++) {
                       (*image)[x*imageSize + i][y*imageSize + j] = png::rgb_pixel(imagearray[i][j] * 255 / maxvalue, imagearray[i][j] * 255 / maxvalue, imagearray[i][j] * 255 / maxvalue);
                    }
                }

            }
         }
        remove(filename.c_str());
        image->write(filename);
        delete image;
    }

    static void writeImagesToPng(std::string filename, float const*images, int numImages, int imageSize) {
        int cols = sqrt(numImages);
        if(cols * cols < numImages) {
            cols++;
        }
        int rows = (numImages + cols - 1) / cols;
        std::cout << "numImages " << numImages << " rows " << rows << " cols " << cols << std::endl;
        png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >(imageSize * rows, imageSize * cols);


        for(int x = 0; x < cols; x++) {
           for(int y = 0; y < rows; y++) {
                if(x * rows + y >= numImages) {
                    continue;
                }
//                cout << "image at x " << x << " y " << y << endl;
                float const*imagearray = &(images[imageSize * imageSize * (x*rows + y) ]);
                float maxValue = getImageMax(imagearray, imageSize);
                float minValue = getImageMin(imagearray, imageSize);
                for(int i = 0; i < imageSize; i++) {
                    for(int j = 0; j < imageSize; j++) {
                       float normValue = (imagearray[i*imageSize + j] + minValue) * 255.0f / (maxValue - minValue);
                       (*image)[x*imageSize + i][y*imageSize + j] = png::rgb_pixel(normValue, normValue, normValue);
                    }
                }

            }
         }
        remove(filename.c_str());
        image->write(filename);
        delete image;
    }
    static void writeImagesToPng(std::string filename, unsigned char const*images, int numImages, int imageSize) {
        int cols = sqrt(numImages);
        if(cols * cols < numImages) {
            cols++;
        }
        int rows = (numImages + cols - 1) / cols;
        std::cout << "numImages " << numImages << " rows " << rows << " cols " << cols << std::endl;
        png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >(imageSize * rows, imageSize * cols);


        for(int x = 0; x < cols; x++) {
           for(int y = 0; y < rows; y++) {
                if(x * rows + y >= numImages) {
                    continue;
                }
//                cout << "image at x " << x << " y " << y << endl;
                unsigned char const*imagearray = &(images[imageSize * imageSize * (x*rows + y) ]);
                float maxValue = 255; // getImageMax(image, imageSize);
                float minValue = 0; // getImageMin(image, imageSize);
                for(int i = 0; i < imageSize; i++) {
                    for(int j = 0; j < imageSize; j++) {
                       float normValue = (imagearray[i*imageSize + j] + minValue) * 255.0f / (maxValue - minValue);
                       (*image)[x*imageSize + i][y*imageSize + j] = png::rgb_pixel(normValue, normValue, normValue);
                    }
                }

            }
         }
        remove(filename.c_str());
        image->write(filename);
        delete image;
    }
};

