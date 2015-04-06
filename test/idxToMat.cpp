// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


// converts mnist idx format files to norb .mat format files

#include <iostream>
#include <stdexcept>

#include "NorbLoader.h"
#include "MnistLoader.h"

using namespace std;

void go( string dir, string setName ) {
    // string matDir = argv[2];
    
    string idxDatfilename = dir + "/" + setName + "-images-idx3-ubyte";
    string idxCatfilename = dir + "/" + setName + "-labels-idx1-ubyte";

    string matDatFilename = dir + "/" + setName + "-dat.mat";
    string matCatFilename = dir + "/" + setName + "-cat.mat";

    int Nimages;
    int Nlabels;
    int imageSize;
    int ***images = MnistLoader::loadImages( dir, setName, &Nimages, &imageSize );
    int *labels = MnistLoader::loadLabels( dir, setName, &Nlabels );
    if( Nimages != Nlabels ) {
         throw runtime_error("mismatch between number of images, and number of labels " + toString(Nimages ) + " vs " +
             toString(Nlabels ) );
    }

    int totalLinearSize = Nimages * imageSize * imageSize;
    unsigned char *imagesUchar = new unsigned char[ totalLinearSize ];
    for( int i = 0; i < totalLinearSize; i++ ) {
        imagesUchar[i] = ( &(images[0][0][0]) )[i];
    }

    NorbLoader::writeLabels( matCatFilename, labels, Nimages );
    NorbLoader::writeImages( matDatFilename, imagesUchar, Nimages, 1, imageSize );

    delete[] imagesUchar;
}

int main( int argc, char *argv[] ) {
    if( argc != 3  ) {
        cout << "Usage: " << argv[0] << " [directory path] [set name, eg t10k or train]" << endl;
        return -1;
    }

    string dir = argv[1];
    string setName = argv[2];

    try {
        go( dir, setName );
    } catch( runtime_error e ) {
        cout << "Something went wrong: " << e.what() << endl;
        return -1;
    }

    return 0;
}


