// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// expected defines:
// one of: [ TANH | RELU | LINEAR ]
// BIASED (or not)

#ifdef TANH
    #define ACTIVATION_FUNCTION(output) (tanh(output))
#elif defined SCALEDTANH
    #define ACTIVATION_FUNCTION(output) (1.7159f * tanh(0.66667f * output))
#elif SIGMOID
    #define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))
#elif defined RELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
#elif defined LINEAR
    #define ACTIVATION_FUNCTION(output) (output)
#endif

void kernel convolve_ints(global const int *p_imageSize, global const int *p_filterSize,
      global const int *image, global const int *filter, global int *result) {
    int id = get_global_id(0);
    int imageSize = p_imageSize[0];
    int filterSize = p_filterSize[0];
    int imageOffset = id / (imageSize * imageSize) * (imageSize * imageSize);
    int localid = id % (imageSize * imageSize);
    int row = localid / imageSize;
    int col = localid % imageSize;
    int halfFilterSize = filterSize >> 1;
    int sum = 0;
    int minm = max(-halfFilterSize, -row);
    int maxm = min(halfFilterSize, imageSize - 1 - row);
    int minn = max(-halfFilterSize, -col);
    int maxn = min(halfFilterSize, imageSize - 1 - col);
    int m = minm;
    while(m <= maxm) {
        int x = (row + m);
        int ximage = imageOffset + x * imageSize;
        int filterrowoffset = (m+halfFilterSize) * filterSize + halfFilterSize;
        int n = minn;
        while(n <= maxn) {
            int y = col + n;
            sum += image[ ximage + y] * filter[ filterrowoffset + n ];
            n++;
        }
        m++;
    }
    result[id] = sum;
}

void kernel convolve_floats(global const int *p_imageSize, global const int *p_filterSize,
      global const float *image, global const float *filter, global float *result) {
    int id = get_global_id(0);
    int imageSize = p_imageSize[0];
    int filterSize = p_filterSize[0];
    int imageOffset = id / (imageSize * imageSize) * (imageSize * imageSize);
    int localid = id % (imageSize * imageSize);
    int row = localid / imageSize;
    int col = localid % imageSize;
    int halfFilterSize = filterSize >> 1;
    float sum = 0;
    int minm = max(-halfFilterSize, -row);
    int maxm = min(halfFilterSize, imageSize - 1 - row);
    int minn = max(-halfFilterSize, -col);
    int maxn = min(halfFilterSize, imageSize - 1 - col);
    int m = minm;
    while(m <= maxm) {
        int x = (row + m);
        int ximage = imageOffset + x * imageSize;
        int filterrowoffset = (m+halfFilterSize) * filterSize + halfFilterSize;
        int n = minn;
        while(n <= maxn) {
            int y = col + n;
            sum += image[ ximage + y] * filter[ filterrowoffset + n ];
            n++;
        }
        m++;
    }
    result[id] = sum;
}

void kernel convolve_imagecubes_int(global const int *p_numInputPlanes, global const int *p_numFilters, 
      global const int *p_imageSize, global const int *p_filterSize,
      global const int *images, global const int *filters, global int *output) {
    int globalId = get_global_id(0);

    int numInputPlanes = p_numInputPlanes[0];
    int numFilters = p_numFilters[0];
    int imageSize = p_imageSize[0];
    int filterSize = p_filterSize[0];
    int imageSizeSquared = imageSize * imageSize;

    int outputImage2Id = globalId / imageSizeSquared;
    int filterId = outputImage2Id % numFilters;
    int inputImage3Id = outputImage2Id / numFilters;

    int filterOffset = filterId * filterSize * filterSize;
    int inputImage3Offset = inputImage3Id * numInputPlanes * imageSizeSquared;

    // intraimage coords
    int localid = globalId % imageSizeSquared;
    int row = localid / imageSize;
    int col = localid % imageSize;

    int halfFilterSize = filterSize >> 1;
    int sum = 0;
    int minm = max(-halfFilterSize, -row);
    int maxm = min(halfFilterSize, imageSize - 1 - row);
    int minn = max(-halfFilterSize, -col);
    int maxn = min(halfFilterSize, imageSize - 1 - col);
    int plane = 0;
    while(plane < numInputPlanes) {
        int inputImageOffset = inputImage3Offset + plane * imageSizeSquared;
        int filterPlaneOffset = filterOffset + plane * filterSize * filterSize;
        int m = minm;
        while(m <= maxm) {
            int y = row + m;
            int inputimagerowoffset = inputImageOffset + y * imageSize;
            int filterrowoffset = filterPlaneOffset + (m+halfFilterSize) * filterSize + halfFilterSize;
            int n = minn;
            while(n <= maxn) {
                int x = col + n;
                sum += images[ inputimagerowoffset + x] * filters[ filterrowoffset + n ];
                n++;
            }
            m++;
        }
        plane++;
    }
    output[globalId] = sum;
}

// receive images as a stack of images
// globalid = n * numfilters * imagesize * imagesize + filter * imagesize * imagesize + imagerow * imagesize + imagecol
//                                 globalid              globalid
//  inputimage3 1 inputimage2 1----filter 1             -> outputimage2 1   outputimage3 1
//                inputimage2 2_/\_filter 2             -> outputimage2 2
//  inputimage3 2 inputimage2 3    filter 1             -> outputimage2 3   outputimage3 2
//                inputimage2 4    filter 2             -> outputimage2 4
//
// each outputimage is only written once, by a combination of:
// - one inputimage3
// - one filter
// each inputimage3 is mapped to each filter once, each time writing to one outputimage
//
// images is:
//       numimages * numinputplanes * imagesizesquared
// filters is:
//       numfilters * numinputplanes * filtersizesquared
// outputs is:
//       numimages * numfilters * outputimagesizesquared

// images are organized like [imageId][plane][row][col]
// filters are organized like [filterid][plane][filterrow][filtercol]
// output are organized like [imageid][filterid][row][col]
void kernel convolve_imagecubes_float( 
      const int numInputPlanes, const int numFilters, 
      const int imageSize, const int filterSize,
      global const float *images, global const float *filters, global float *output) {
    int globalId = get_global_id(0);

    int imageSizeSquared = imageSize * imageSize;

    int outputImage2Id = globalId / imageSizeSquared;
    int filterId = outputImage2Id % numFilters;
    int inputImage3Id = outputImage2Id / numFilters;

    int filterOffset = filterId * filterSize * filterSize;
    int inputImage3Offset = inputImage3Id * numInputPlanes * imageSizeSquared;

    // intraimage coords
    int localid = globalId % imageSizeSquared;
    int row = localid / imageSize;
    int col = localid % imageSize;

    int halfFilterSize = filterSize >> 1;
    float sum = 0;
    // m should vary from -halfFilterSize through 0 to halfFilterSize 
    // n too...
    int minm = max(-halfFilterSize, -row);
    int maxm = min(halfFilterSize, imageSize - 1 - row);
    int minn = max(-halfFilterSize, -col);
    int maxn = min(halfFilterSize, imageSize - 1 - col);
    int inputPlane = 0;
    while(inputPlane < numInputPlanes) {
        int inputImageOffset = inputImage3Offset + inputPlane * imageSizeSquared;
        int m = minm;
        while(m <= maxm) {
            int y = row + m;
            int inputimagerowoffset = inputImageOffset + y * imageSize;
            int filterrowoffset = filterOffset + (m+halfFilterSize) * filterSize + halfFilterSize;
            int n = minn;
            while(n <= maxn) {
                int x = col + n;
                sum += images[ inputimagerowoffset + x] * filters[ filterrowoffset + n ];
                n++;
            }
            m++;
        }
        inputPlane++;
    }

    output[globalId] = sum;
}

void kernel convolve_imagecubes_float_nopadzeros( 
      const int numInputPlanes, const int numFilters, 
      const int inputSize, const int filterSize,
      global const float *images, global const float *filters, global float *output) {
    int globalId = get_global_id(0);

    int inputSizeSquared = inputSize * inputSize;
    int outputSize = inputSize - filterSize + 1;
    int outputSizeSquared = outputSize * outputSize;

    int outputImage2Id = globalId / outputSizeSquared;
    int filterId = outputImage2Id % numFilters;
    int inputImage3Id = outputImage2Id / numFilters;

    int filterOffset = filterId * filterSize * filterSize;
    int inputImage3Offset = inputImage3Id * numInputPlanes * inputSizeSquared;

    // intraimage coords
    int localid = globalId % outputSizeSquared;
    int outputRow = localid / outputSize;
    int outputCol = localid % outputSize;

    int halfFilterSize = filterSize >> 1;
    float sum = 0;
    int minm = -halfFilterSize;
    int maxm = halfFilterSize;
    int minn = -halfFilterSize;
    int maxn = halfFilterSize;
    int inputPlane = 0;
    while(inputPlane < numInputPlanes) {
        int inputImageOffset = inputImage3Offset + inputPlane * inputSizeSquared;
        int m = minm;
        while(m <= maxm) {
            int inputRow = outputRow + m + halfFilterSize;
            int inputimagerowoffset = inputImageOffset + inputRow * inputSize;
            int filterrowoffset = filterOffset + (m+halfFilterSize) * filterSize + halfFilterSize;
            int n = minn;
            while(n <= maxn) {
                int inputCol = outputCol + n + halfFilterSize;
                sum += images[ inputimagerowoffset + inputCol] * filters[ filterrowoffset + n ];
                n++;
            }
            m++;
        }
        inputPlane++;
    }
    output[globalId] = sum;
}

