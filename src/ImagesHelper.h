// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>

#include "ImageHelper.h"

class ImagesHelper {
public:
    static int ***allocateImages( int N, int imageSize ) {
        int *contiguousSpace = new int[N * imageSize * imageSize ];
        if( contiguousSpace == 0 ) {
            throw std::runtime_error("failed to allocate memory");
        }
        int ***images = new int**[N];
        if( images == 0 ) {
            throw std::runtime_error("failed to allocate int***images memory");
        }
        for( int n = 0; n < N; n++ ) {
           int **image = new int*[imageSize];
           int *thisimagecontiguousspace = &(contiguousSpace[n * imageSize * imageSize]);
            if( image == 0 ) {
                throw std::runtime_error("failed to allocate int **image memory");
            }
           images[n] = image;
           for( int i = 0; i < imageSize; i++ ) {
              image[i] = &(thisimagecontiguousspace[i * imageSize ]);
           }
        }

//        int ***images = new int**[N];
//        for( int i = 0; i < N; i++ ) {
//            images[i] = ImageHelper::allocateImage( imageSize );
//        }
        return images;
    }

    static void deleteImages( int ****p_images, int N, int imageSize ) {
        int ***images = *p_images;
        int *contiguous = &(images[0][0][0] );
        for( int n = 0; n < N; n++ ) {
            delete[] images[n];
        }
        delete[] images;       
        delete[] contiguous;

//        for( int n = 0; n < N; n++ ) {
//            ImageHelper::deleteImage( &(*images)[n], imageSize );
//        }
//        delete[] (*images);
        *p_images = 0;
    }

    static float ***allocateImagesFloats( int N, int imageSize ) {
        float *contiguousSpace = new float[N * imageSize * imageSize ];
        if( contiguousSpace == 0 ) {
            throw std::runtime_error("failed to allocate memory");
        }
        float ***images = new float**[N];
        if( images == 0 ) {
            throw std::runtime_error("failed to allocate int***images memory");
        }
        for( int n = 0; n < N; n++ ) {
           float **image = new float*[imageSize];
           float *thisimagecontiguousspace = &(contiguousSpace[n * imageSize * imageSize]);
            if( image == 0 ) {
                throw std::runtime_error("failed to allocate int **image memory");
            }
           images[n] = image;
           for( int i = 0; i < imageSize; i++ ) {
              image[i] = &(thisimagecontiguousspace[i * imageSize ]);
           }
        }

//        int ***images = new int**[N];
//        for( int i = 0; i < N; i++ ) {
//            images[i] = ImageHelper::allocateImage( imageSize );
//        }
        return images;
    }

    static void printImages( float ***images, int N, int imageSize ) {
        for( int n = 0; n < N; n++ ) {
            std::cout << "image " << n << std::endl;
            ImageHelper::print( images[n], imageSize );
        }
    }

    static void deleteImages( float ****p_images, int N, int imageSize ) {
        float ***images = *p_images;
        float *contiguous = &(images[0][0][0] );
        for( int n = 0; n < N; n++ ) {
            delete[] images[n];
        }
        delete[] images;       
        delete[] contiguous;

//        for( int n = 0; n < N; n++ ) {
//            ImageHelper::deleteImage( &(*images)[n], imageSize );
//        }
//        delete[] (*images);
        *p_images = 0;
    }

    static void copyImages( float ***target, int ***source, int N, int imageSize ) {
        for( int n = 0; n < N; n++ ) {
           ImageHelper::copyImage( target[n], source[n], imageSize );
        }
//        float *target1d = &(target[0][0][0]);
//        int *source1d = &(source[0][0][0]);
//        const int T = N * imageSize * imageSize;
//        for( int i = 0; i < T; i++ ) {
//            target1d[i] = source1d[i];
//        }
    }
};

