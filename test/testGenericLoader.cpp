#include <iostream>
#include <string>

//#include "NorbLoader.h"
#include "ImagePng.h"
#include "GenericLoader.h"
#include "NormalizationHelper.h"

using namespace std;

void go( string trainFilepath, int startN, int numExamples ) {
    int N;
    int numPlanes;
    int imageSize;
//    int totalSize;
    GenericLoader::getDimensions( trainFilepath, &N, &numPlanes, &imageSize );
    cout << "N " << N << " numplanes " << numPlanes << " imageSize " << imageSize << endl;
    unsigned char *images = new unsigned char[ numExamples * numPlanes * imageSize * imageSize ];
    int *labels = new int[ numExamples ];
    GenericLoader::load( trainFilepath, images, labels, startN, numExamples );
//    float *images = new float[ N * numPlanes * imageSize * imageSize ];
//    for( int i = 0; i < N * numPlanes * imageSize * imageSize; i++ ) {
//        images[i] = imagesUchar[i];
//    }
    float thismin;
    float thismax;
    NormalizationHelper::getMinMax( images, numExamples * numPlanes * imageSize * imageSize, &thismin, &thismax );
    cout << "min: " << thismin << " max: " << thismax << endl;
    ImagePng::writeImagesToPng( "testGenericLoader.png", images, numExamples * numPlanes, imageSize );
    for( int i = 0; i < numExamples; i++ ) {
        cout << "labels[" << i << "]=" << labels[i] << endl;
    }
//    float *translated = new float[N * numPlanes * imageSize * imageSize];
//    Translator::translate( n, numPlanes, imageSize, translateRows, translateCols, images, translated );
//    ImagePng::writeImagesToPng( "testTranslator-2.png", translated + n * numPlanes * imageSize * imageSize, numPlanes, imageSize );
}

int main( int argc, char *argv[] ) {
    if( argc != 4 ) {
        cout << "Usage: [trainfilepath] [startn] [numexamples]" << endl;
        return -1;
    }
    string trainFilepath = string( argv[1] );
    int startN = atoi( argv[2] );
    int numExamples = atoi( argv[3] );
    go( trainFilepath, startN, numExamples );
    return 0;
}


