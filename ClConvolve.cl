// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

////    if( globalId == 0 ) {
////        for( int i = 0; i < 4; i++ ) {
////            results[14 + i] = thisOffset;
////        }
////    }
//    if( globalId == 0 ) {
//        for( int i = 0; i < gUpstreamBoardSizeSquared; i++ ) {
//            results[100 * (1+upstreamPlane) + i] = _upstreamBoard[i];
//        }
//    }
//    if( globalId == 12 ) {
////        results[400 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] = sum;
////        results[400 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] = _upstreamBoard[ inputboardrowoffset + inputCol];
//        results[400 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] = inputboardrowoffset + inputCol;
////        results[400 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] = minu;
////        results[400 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] += 1;
//        results[600 + 100 * upstreamPlane + (u+2) * 5 + (v+2) ] = _filterCube[ filterrowoffset + v ];
//    }
//    if( globalId == 0 ) {
//        for( int i = 0; i < filterCubeLength; i++ ) {
//            results[300 + i] = _filterCube[i];
//        }
//    }
////    if( globalId == 12 ) {
////        results[500 + 0] = 
////    }

////    results[globalId*2] = images[25+globalId];
////    results[globalId*2+1] = _upstreamBoard[globalId];


kernel void add_in_place( const int N, global const float*in, global float*target ) {
    int globalId = get_global_id(0);
    if( globalId < N ) {
        target[globalId] += in[globalId];
    }
}


