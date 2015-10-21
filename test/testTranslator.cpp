#include <iostream>
#include <string>

#include "loaders/NorbLoader.h"
#include "util/ImagePng.h"
#include "patches/Translator.h"

using namespace std;

void go(string dataDir, string setName, int n, int translateRows, int translateCols) {
    int N;
    int numPlanes;
    int imageSize;
    string filePath = dataDir + "/" + setName + "-dat.mat";
    NorbLoader::getDimensions(filePath, &N, &numPlanes, &imageSize);
    N = n + 1;
    unsigned char *imagesUchar = new unsigned char[ N * numPlanes * imageSize * imageSize ];
    int *labels = new int[ N ];
    NorbLoader::load(filePath, imagesUchar, labels, 0, N);
    cout << "n " << n << " N " << N << endl;
    float *images = new float[ N * numPlanes * imageSize * imageSize ];
    for(int i = 0; i < N * numPlanes * imageSize * imageSize; i++) {
        images[i] = imagesUchar[i];
    }
    ImagePng::writeImagesToPng("testTranslator-1.png", images + n * numPlanes * imageSize * imageSize, numPlanes, imageSize);
    float *translated = new float[N * numPlanes * imageSize * imageSize];
    Translator::translate(n, numPlanes, imageSize, translateRows, translateCols, images, translated);
    ImagePng::writeImagesToPng("testTranslator-2.png", translated + n * numPlanes * imageSize * imageSize, numPlanes, imageSize);
}

int main(int argc, char *argv[]) {
    if(argc != 6) {
        cout << "Usage: [datadir] [setname] [n] [translaterows] [translatecols]" << endl;
        return -1;
    }
    string dataDir = string(argv[1]);
    string setName = string(argv[2]);
    int n = atoi(argv[3]);
    int translateRows = atoi(argv[4]);
    int translateCols = atoi(argv[5]);
    go(dataDir, setName, n, translateRows, translateCols);
    return 0;
}


