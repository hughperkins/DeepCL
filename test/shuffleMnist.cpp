#include <iostream>
using namespace std;

int main(int argc, char *argv[] ) {
    string mnistDir = argv[1];
    string inSet = argv[2];
    string outSet = argv[3];
    int N;
    int boardSize;
    int ***images = MnistLoader::loadImages( mnistDir, inSet, &N, &boardSize );
    random_shuffle( images, images + N );
    MnistLoader::writeImages( images, mnistDir, outSet, N, boardSize );
    return 0;
}

