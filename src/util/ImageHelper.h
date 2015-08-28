// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

class ImageHelper {
public:
//static int **allocateImage(int imageSize) {
    //int **image = new int*[imageSize];
//    for(int i = 0; i < imageSize; i++) {
//        image[i] = new int[imageSize];
//        for(int j = 0; j < imageSize; j++) {
//            image[i][j] = 0;
//        }
//    }
//    int *contiguousarray = new int[ imageSize * imageSize ];
//    memset(contiguousarray, 0, sizeof(int) * imageSize * imageSize);
//    int **image = new int*[imageSize];
//    for(int i = 0; i < imageSize; i++) {
//        image[i] = &(contiguousarray[i*imageSize]);
//    }
//    return image;
//}

//static float **allocateFloats(int imageSize) {
//    //int **image = new int*[imageSize];
////    for(int i = 0; i < imageSize; i++) {
////        image[i] = new int[imageSize];
////        for(int j = 0; j < imageSize; j++) {
////            image[i][j] = 0;
////        }
////    }
//    float *contiguousarray = new float[ imageSize * imageSize ];
//    memset(contiguousarray, 0, sizeof(float) * imageSize * imageSize);
//    float **image = new float*[imageSize];
//    for(int i = 0; i < imageSize; i++) {
//        image[i] = &(contiguousarray[i*imageSize]);
//    }
//    return image;
//}

//static void deleteImage(int ***p_image, int imageSize) {
//   if(p_image == 0) {
//      return;
//   }
//   delete[] (*p_image)[0];
//   delete[] *p_image;
//   *p_image = 0;
//}

//static void deleteImage(float ***p_image, int imageSize) {
//   if(p_image == 0) {
//      return;
//   }
//   delete[] (*p_image)[0];
//   delete[] *p_image;
//   *p_image = 0;
//}

//static void copyImage(int *const*const dst, int const*const *const src, int imageSize) {
//    for(int i = 0; i < imageSize; i++) {
//        for(int j = 0; j < imageSize; j++) {
//            if(dst[i][j] != src[i][j]) {
//                dst[i][j] = src[i][j];
//            }
//        }
//    }
//}

//static void copyImage(float *const*const dst, int const*const *const src, int imageSize) {
//    for(int i = 0; i < imageSize; i++) {
//        for(int j = 0; j < imageSize; j++) {
//            if(dst[i][j] != src[i][j]) {
//                dst[i][j] = src[i][j];
//            }
//        }
//    }
//}

//static void wipeImage(int *const*const image, int imageSize) {
//    for(int i = 0; i < imageSize; i++) {
//        for(int j = 0; j < imageSize; j++) {
//            image[i][j] = 0;
//        }
//    }
//}

//static void printInts(int const*const*const image, int imageSize) {
//    std::ostringstream ss;
//    ss << "\n";
//    for(int i = 0; i < imageSize; i++) {
//       for(int j = 0; j < imageSize; j++) {
//          ss << image[i][j] << " ";
//       }
//       ss << "\n";
//    }
//    std::cout << ss.str() << std::endl;
//}

//static void print(float const*const*const image, int imageSize) {
//    std::ostringstream ss;
//    ss << "\n";
//    for(int i = 0; i < imageSize; i++) {
//       for(int j = 0; j < imageSize; j++) {
//          ss << image[i][j] << " ";
//       }
//       ss << "\n";
//    }
//    std::cout << ss.str() << std::endl;
//}

static void _printImage(int *image, int imageSize) {
    std::ostringstream ss;
    ss << "\n";
    for(int i = 0; i < imageSize; i++) {
        for(int j = 0; j < imageSize; j++) {
            int offset = i * imageSize + j;
            int value = image[offset];
            if(value == 0) {
                ss << ".";
            }
            if(value == 1) {
                ss << "*";
            }
            if(value == 2) {
                ss << "O";
            }
            if(value == 3) {
                ss << "+";
            }
        }
        ss << "\n";
    }
    std::cout << ss.str() << std::endl;
}

static void _printImage(float *image, int imageSize) {
    std::ostringstream ss;
    ss << "\n";
    for(int i = 0; i < imageSize; i++) {
        for(int j = 0; j < imageSize; j++) {
            int offset = i * imageSize + j;
            float value = image[offset];
            if(value == 0) {
                ss << ".";
            }
            if(value == 1) {
                ss << "*";
            }
            if(value == 2) {
                ss << "O";
            }
            if(value == 3) {
                ss << "+";
            }
        }
        ss << "\n";
    }
    std::cout << ss.str() << std::endl;
}

//static void printImage(int const *const *const image, int imageSize) {
///*    int numdigits = 1;
//    for(int i = 0; i < imageSize; i++) {
//       for(int j = 0; j < imageSize; j++) {
//          std::string thisnum = toString(image[i][j]);
//          int thisdigits = thisnum.length();
//          numdigits = thisdigits > numdigits ? thisdigits : numdigits;
//       }
//    }*/
//    ostringstream ss;
//    ss << "\n";
//    for(int i = 0; i < imageSize; i++) {
//       for(int j = 0; j < imageSize; j++) {
//          if(image[i][j] == 0) {
//              ss << ".";
//          }
//          if(image[i][j] == 1) {
//              ss << "*";
//          }
//          if(image[i][j] == 2) {
//              ss << "O";
//          }
//          if(image[i][j] == 3) {
//              ss << "+";
//          }
//       }
//       ss << "\n";
//    }
//    debug(ss.str());
//}

//static int **loadImage(std::string filepath, int *p_imageSize) {
//    std::ifstream f;
//    f.open(filepath.c_str());
//    //f >> imageSize;
//    //int **image = 0;
//   std::string thisline;
//   f >> thisline;
//   *p_imageSize = (int)thisline.length();
//   if(*p_imageSize == 0) {
//      std::cout << "imagehelper::loadImage. error: imagesize 0, " << filepath << std::endl;
//      throw "imagehelper::loadImage. error: imagesize 0 " + filepath;
//   }
//   //cout << "imagesize: " << imageSize << std::endl;
//   int **image = allocateImage(*p_imageSize);
//    for(int i = 0; i < *p_imageSize; i++) {
//       if(i == 0) {
//       }
//       for(int j = 0; j < *p_imageSize; j++) {
//          std::string thischar = std::string("") + thisline[j];
//          if(thischar == "*") {
//              image[i][j] = 1;
////              (*p_piecesPlaced)++;
//          }
//          if(thischar == "O") {
//              image[i][j] = 2;
////              (*p_piecesPlaced)++;
//          }
//       }
//       f >> thisline;
//    }
//    f.close();
//    return image;
//}
};


