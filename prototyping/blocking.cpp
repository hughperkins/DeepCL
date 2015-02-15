// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cmath>
using namespace std;

#include "stringhelper.h"

class Dimensions {
public:
    int inputPlanes;
    int numFilters;
    int filterSize;
    int filterSizeBytes;
    int filterPlanesPerWorkgroup;
};

ostream &operator<<( ostream &os, Dimensions const &dim ) {
    os << "Dimensions{ inputPlanes=" << dim.inputPlanes << " numFilters=" << dim.numFilters
        << " filterSize=" << dim.filterSize << " filterSizeBytes=" << dim.filterSizeBytes
        << " filterPlanesPerWorkgroup=" << dim.filterPlanesPerWorkgroup << " }";
    return os;
}

int calcCost( Dimensions &dim, int inPerBlock, int outPerBlock ) {
    int numInBlocks = ( dim.inputPlanes + inPerBlock - 1 ) / inPerBlock;
    int numOutBlocks = ( dim.numFilters + outPerBlock - 1 ) / outPerBlock;
    int numWorkgroups = numInBlocks * numOutBlocks;
    int costPerWorkgroup = // = number of input and output planes to load into local memory?
        numInBlocks + numOutBlocks;
//    cout << "  numInBlocks=" << numInBlocks << " numOutBlocks=" << numOutBlocks 
//        << " numWorkgroups=" << numWorkgroups << " costperworkgroup=" << costPerWorkgroup << endl;
    int totalCost = numWorkgroups * costPerWorkgroup;
    cout << "  numInBlocks=" << numInBlocks << " numOutBlocks=" << numOutBlocks 
        << " numWorkgroups=" << numWorkgroups << " costperworkgroup=" << costPerWorkgroup
        << " total=" << totalCost << endl;
    return totalCost;
}

int main( int argc, char *argv[] ) {
    Dimensions dim;
    dim.inputPlanes = atoi( argv[1] );
    dim.numFilters = atoi( argv[2] );
    dim.filterSize = atoi( argv[3] );
    int maxWorkgroupSize = atoi( argv[4] );
    dim.filterSizeBytes = dim.filterSize * dim.filterSize * 4;
    dim.filterPlanesPerWorkgroup = maxWorkgroupSize / dim.filterSize / dim.filterSize;
//    cout << "inputplanes=" << inputPlanes << " numFilters=" << numFilters << " filterSize=" << filterSize <<
//        " filterSizeBytes=" << filterSizeBytes << " filterPlanesPerWorkgroup=" << filterPlanesPerWorkgroup << endl;
//    vector< pair< int, int > > ifPairs;
//    for( int i = 0; i < inputPlanes; i++ ) {
//        for( int f = 0; f < numFilters; f++ ) {
//            ifPairs.push_back( pair< int, int >( i, f ) );
//        }
//    }
    cout << dim << endl;

    int bestInPerBlock = -1;
    int bestOutPerBlock = -1;
    int bestCost = 0;
    for( int inPerBlock = 1; inPerBlock <= dim.inputPlanes; inPerBlock++ ) {
        if( inPerBlock > dim.filterPlanesPerWorkgroup ) {
            break;
        }
        int outPerBlock = dim.filterPlanesPerWorkgroup / inPerBlock;
        cout << "inperblock=" << inPerBlock << " outperblock=" << outPerBlock << endl;
        int cost = calcCost( dim, inPerBlock, outPerBlock );
        if( bestInPerBlock == -1 || cost < bestCost ) {
            bestCost = cost;
            bestInPerBlock = inPerBlock;
            bestOutPerBlock = outPerBlock;
        }
    }
    cout << "best blocking: inperblock=" << bestInPerBlock << " outperblock=" << bestOutPerBlock << " cost=" << bestCost << endl;

//    int blockRows = floor( sqrt( filterPlanesPerWorkgroup ) );
//    int blockCols = ( filterPlanesPerWorkgroup + blockRows - 1 ) / blockRows;
//    int blockLocalRows = 
//    int numWorkgroups = ( inputPlanes * numFilters + filterPlanesPerWorkgroup - 1 ) / filterPlanesPerWorkgroup;
//    cout << "numWorkgroups=" << numWorkgroups << " blockRows=" << blockRows << " blockCols=" << blockCols << endl;
//    for( int w = 0; w < numWorkgroups; w++ ) {
////        cout << "workgroup: " << w << endl;
//        int blockRow = w / blockCols;
//        int blockCol = w % blockCols;
//        cout << "workgroup=" << w << " blockRow=" << blockRow << " blockCol=" << blockCol << endl;
////        int ifoffset = w * filterPlanesPerWorkgroup;
////        for( int i = 0; i < filterPlanesPerWorkgroup; i++ ) {
////            //cout << ifPairs[ ifoffset + i].first << ","  << ifPairs[ ifoffset + i].second << endl;
////        }
//    }
}

// eg layers of 32c5
// 32 x 32, current scratch has 1024 workgroups
// new blocking has 51 workgroups
// each workgroup has 


