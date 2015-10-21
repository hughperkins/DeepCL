#include <iostream>
#include <string>

#include "loaders/NorbLoader.h"
#include "util/ImagePng.h"
#include "patches/PatchExtractor.h"

using namespace std;

void go(string dataDir, string setName, int n, int patchSize, int patchRow, int patchCol) {
    int N;
    int numPlanes;
    int imageSize;
    unsigned char *imagesUchar = NorbLoader::loadImages(dataDir + "/" + setName + "-dat.mat", &N, &numPlanes, &imageSize, n + 1);
    cout << "n " << n << " N " << N << endl;
    N = n + 1;
    float *images = new float[ N * numPlanes * imageSize * imageSize ];
    for(int i = 0; i < N * numPlanes * imageSize * imageSize; i++) {
        images[i] = imagesUchar[i];
    }
    ImagePng::writeImagesToPng("testPatchExtractor-1.png", images + n * numPlanes * imageSize * imageSize, numPlanes, imageSize);
    float *patches = new float[N * numPlanes * patchSize * patchSize];
    PatchExtractor::extractPatch(n, numPlanes, imageSize, patchSize, patchRow, patchCol, images, patches);
    ImagePng::writeImagesToPng("testPatchExtractor-2.png", patches + n * numPlanes * patchSize * patchSize, numPlanes, patchSize);
}

int main(int argc, char *argv[]) {
    if(argc != 7) {
        cout << "Usage: [datadir] [setname] [n] [patchsize] [patchrow] [patchcol]" << endl;
        return -1;
    }
    string dataDir = string(argv[1]);
    string setName = string(argv[2]);
    int n = atoi(argv[3]);
    int patchSize = atoi(argv[4]);
    int patchRow = atoi(argv[5]);
    int patchCol = atoi(argv[6]);
    go(dataDir, setName, n, patchSize, patchRow, patchCol);
    return 0;
}


