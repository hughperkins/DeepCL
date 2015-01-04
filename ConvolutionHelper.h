#pragma once

#include <stdexcept>
#include <iostream>



//template<typename T>
//void arraySet( int size, T*array, T value ) {
//    for( int i = 0; i < size; i++ ) {
//        array[i] = value;
//    }
//}

//class ConvolutionHelper {
//public:
//    // we expect that din1, din2, dw each have 5 values
//    // zeropad has same number of values as dimensions, ie 5
//    // 1 means zeropad that dimension (so dout[d] == max(din1[d],din2[d]))
//    // 0 means no zeropad (so dout[d] == abs(din[d]-dw[d])+1 )
//    static void convolveCpu( const int D, 
//        int const* din1, int const *din2, int *dout, 
//        int *zeroPad,
//        float const* in1, float const *in2, float *out ) {
//        if( D != 5 ) {
//            throw std::runtime_error("D must be 5 :-)" );
//        }
//        for( int d = 0; d < 5; d++ ) {
//            if( zeroPad[d] != 0 ) {
//                throw std::runtime_error("zeroPad must be 0 :-D  Just to simplify, no theoretical reason why" );
//            }
//            dout[d] = abs( din1[d] - din2[d] ) + 1;
//        }
//        int in1pos[5];
//        int in2pos[5];
//        int outpos[5];
//        arraySet( 5, in1pos, 0 );
//        arraySet( 5, in2pos, 0 );
//        arraySet( 5, outpos, 0 );
//        bool outloopdone = false;
//        bool in1loopdone = false;
//        bool in2loopdone = false;
//        while( !outloopdone ) {
//            float thisresult = 0;
//            while( !in1loopdone ) {
//                while( !in2loopdone ) {
//                }
//            }
//        }
//    }
//};

