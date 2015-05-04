// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class AccuracyHelper{
public:
    static int calcNumRight( int numImages, int numPlanes, int const*labels, float const*output ) {
        int correct = 0;
        for( int n = 0; n < numImages; n++ ) {
            double maxValue = -100000;
            int bestIndex = -1;
            int bestCount = 0;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                if( output[ n * numPlanes + plane ] > maxValue ) {
                    bestIndex = plane;
                    maxValue = output[ n * numPlanes + plane ];
                   bestCount = 1;
                } else if ( output[ n * numPlanes + plane ] == maxValue ) {
                   bestCount++;
                }
            }
//            cout << "expected: " << labels[n] << " got " << bestIndex << endl;
            if( bestIndex == labels[n] && bestCount == 1 ) {
                correct++;
            }
        }
        return correct;
    }
    static void printAccuracy( int numImages, int numPlanes, int const*labels, float const*output ) {
        int correct = 0;
        for( int n = 0; n < numImages; n++ ) {
            double maxValue = -100000;
            int bestIndex = -1;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                if( output[ n * numPlanes + plane ] > maxValue ) {
                    bestIndex = plane;
                    maxValue = output[ n * numPlanes + plane ];
                }
            }
//            cout << "expected: " << labels[n] << " got " << bestIndex << endl;
            if( bestIndex == labels[n] ) {
                correct++;
            }
        }
        std::cout << " accuracy: " << correct << "/" << numImages << " " << ((float)correct * 100.0f / numImages ) << "%" << std::endl;
    }
};

