// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "OpenCLHelper.h"

#include "DropoutPropagate.h"
#include "ActivationFunction.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/WeightRandomizer.h"

using namespace std;

namespace testdropoutpropagate {

//void intsToBits( int numBits, int *ints, unsigned char *bitfield ) {
////    int numBytes = (N+8-1)/8;
////    unsigned char *bitsField = new unsigned char[numBytes];
//    int idx = 0;
//    unsigned char thisByte = 0;
//    int bitsPacked = 0;
//    for( int i = 0; i < numBits; i++ ) {
////        double value = ( (int)random() % 10000 ) / 20000.0f + 0.5f;
//        unsigned char bit = ints[i];
////        unsigned char bit = 0;
//        thisByte <<= 1;
//        thisByte |= bit;
//        bitsPacked++;
//        if( bitsPacked >= 8 ) {
//            bitfield[idx] = thisByte;
//            idx++;
//            bitsPacked = 0;
//        }
//    }
//}

TEST( testdropoutpropagate, basic ) {
    int batchSize = 1;
    int numPlanes = 1;
    int imageSize = 3;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    DropoutPropagate *dropoutPropagate = DropoutPropagate::instanceForTest( cl, numPlanes, imageSize, 0.6f );
    unsigned char mask[] = { 1, 0, 0,
                             0,0,1,
                             1,0,1
    };
//    unsigned char ucMask[2];
//    intsToBits( 8, maskInts, ucMask );
    float data[] = { 1, -2, 5,
                     3, 8.2f, 4.1f,
                     3, -33.1f, 14.2f,
    };
    int resultsSize = dropoutPropagate->getResultsSize( batchSize );
    EXPECT_EQ( resultsSize, imageSize * imageSize );
    float *output = new float[resultsSize];

    dropoutPropagate->propagate( batchSize, mask, data, output );

    EXPECT_EQ( 1 * 5/3, output[0] );
    EXPECT_EQ( 0, output[1] );
    EXPECT_EQ( 0, output[2] );

    EXPECT_EQ( 0, output[3] );
    EXPECT_EQ( 0, output[4] );
    EXPECT_EQ( 4.1f * 5/3, output[5] );

    EXPECT_EQ( 3 * 5/3, output[6] );
    EXPECT_EQ( 0, output[7] );
    EXPECT_EQ( 14.2f * 5/3, output[8] );

    delete dropoutPropagate;
    delete[] output;
    delete cl;
}

TEST( testdropoutpropagate, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int imageSize = 2;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    DropoutPropagate *dropoutPropagate = DropoutPropagate::instanceForTest( cl, numPlanes, imageSize, 0.6f );
    float data[] = { 1, 2, 
                    5, 3,

                     3, 8, 
                    4, 1,

                     3, 33, 
                    14,23,

                     -1, -3.5f,
                    37.4f,5
    };
    unsigned char mask[] = {
        0,1,
        1,0,

        1,1,
        0,0,

        0,0,
        0,1,

        1,1,
        0,1
    };
    int outputSize = dropoutPropagate->getResultsSize( batchSize );
    float *output = new float[outputSize];

    dropoutPropagate->propagate( batchSize, mask, data, output );

    EXPECT_EQ( output[0], 0 );
    EXPECT_EQ( output[1], 2 * 5/3 );
    EXPECT_EQ( output[2], 5 * 5/3 );

    EXPECT_EQ( output[12], -1 * 5/3 );
    EXPECT_EQ( output[13], -3.5f * 5/3 );
    EXPECT_EQ( output[14], 0 * 5/3 );
    EXPECT_EQ( output[15], 5 * 5/3 );

    delete dropoutPropagate;
    delete[] output;
    delete cl;
}

TEST( testdropoutpropagate, fromwrappers ) {
    int batchSize = 1;
    int numPlanes = 1;
    int imageSize = 4;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    DropoutPropagate *dropoutPropagate = DropoutPropagate::instanceForTest( cl, numPlanes, imageSize, 0.6f );
    float input[] = { 1, -2, -5, 3,
                     3, 8, 4, 1,
                     3, 33, 14,23,
                     -1, -3.5f,37.4f,5
    };
    unsigned char mask[] = {
            1,0,0,1,
            0,1,1,0,
            1,0,1,0,
            0,0,1,1
    };
    int outputSize = dropoutPropagate->getResultsSize( batchSize );
    float *output = new float[outputSize];

    const int inputSize = batchSize * numPlanes * imageSize * imageSize;
    CLWrapper *maskWrapper = cl->wrap( inputSize, mask );
    CLWrapper *inputWrapper = cl->wrap( inputSize, input );
    CLWrapper *outputWrapper = cl->wrap( outputSize, output );

    maskWrapper->copyToDevice();
    inputWrapper->copyToDevice();

    dropoutPropagate->propagate( batchSize, maskWrapper, inputWrapper, outputWrapper );

    outputWrapper->copyToHost();

    EXPECT_EQ( output[0], 1 * 5/3 );
    EXPECT_EQ( output[1], 0 );
    EXPECT_EQ( output[2], 0 );
    EXPECT_EQ( output[3], 3 * 5/3 );
    EXPECT_EQ( output[12], 0 );
    EXPECT_EQ( output[13], 0 );
    EXPECT_EQ( output[14], 37.4f * 5/3 );
    EXPECT_EQ( output[15], 5 * 5/3 );

    delete maskWrapper;
    delete inputWrapper;
    delete outputWrapper;
    delete dropoutPropagate;
    delete[] output;
    delete cl;
}

class CompareSpecificArgs{
public:
    static CompareSpecificArgs instance() { 
        static CompareSpecificArgs args; 
        return args; 
    }

    // [[[cog
    // floats= ['dropRatio']
    // ints = ['batchSize', 'numPlanes', 'imageSize', 'instance0', 'instance1' ]
    // strings = []
    // import cog_fluent
    // cog_fluent.gov3( 'CompareSpecificArgs', ints = ints, floats = floats, strings=strings )
    // ]]]
    // generated, using cog:
    int _batchSize;
    int _numPlanes;
    int _imageSize;
    int _instance0;
    int _instance1;
    float _dropRatio;
    CompareSpecificArgs() {
        _batchSize = 0;
        _numPlanes = 0;
        _imageSize = 0;
        _instance0 = 0;
        _instance1 = 0;
        _dropRatio = 0;
    }
    CompareSpecificArgs batchSize( int _batchSize ) {
        this->_batchSize = _batchSize;
        return *this;
    }
    CompareSpecificArgs numPlanes( int _numPlanes ) {
        this->_numPlanes = _numPlanes;
        return *this;
    }
    CompareSpecificArgs imageSize( int _imageSize ) {
        this->_imageSize = _imageSize;
        return *this;
    }
    CompareSpecificArgs instance0( int _instance0 ) {
        this->_instance0 = _instance0;
        return *this;
    }
    CompareSpecificArgs instance1( int _instance1 ) {
        this->_instance1 = _instance1;
        return *this;
    }
    CompareSpecificArgs dropRatio( float _dropRatio ) {
        this->_dropRatio = _dropRatio;
        return *this;
    }
    // [[[end]]]
};

void compareSpecific( CompareSpecificArgs args ) {
    cout << "instance0: " << args._instance0 << endl;
    cout << "instance1: " << args._instance1 << endl;

    int batchSize = args._batchSize;
    int numPlanes = args._numPlanes;
    int imageSize = args._imageSize;

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

    DropoutPropagate *dropoutPropagate0 = DropoutPropagate::instanceSpecific( args._instance0, cl, numPlanes, imageSize, args._dropRatio );
    DropoutPropagate *dropoutPropagate1 = DropoutPropagate::instanceSpecific( args._instance1, cl, numPlanes, imageSize, args._dropRatio );

    const int inputSize = batchSize * numPlanes * imageSize * imageSize;
    int outputSize = dropoutPropagate0->getResultsSize( batchSize );

    unsigned char *mask = new unsigned char[ inputSize ];
    float *input = new float[ inputSize ];
    float *output = new float[ outputSize ];

    CLWrapper *maskWrapper = cl->wrap( inputSize, mask );
    CLWrapper *inputWrapper = cl->wrap( inputSize, input );
    CLWrapper *outputWrapper = cl->wrap( outputSize, output );

    WeightRandomizer::randomizeInts( mask, inputSize, 0, 1 );
    WeightRandomizer::randomize( input, inputSize, -0.1f, 0.1f );

    memset( output, 99, sizeof(int) * outputSize );

    maskWrapper->copyToDevice();
    inputWrapper->copyToDevice();
    outputWrapper->copyToDevice();

    dropoutPropagate0->propagate( batchSize, maskWrapper, inputWrapper, outputWrapper );
    outputWrapper->copyToHost();

    float *output0 = new float[ outputSize ];
    memcpy( output0, output, sizeof(float) * outputSize );
    
    memset( output, 99, sizeof(int) * outputSize );

    maskWrapper->copyToDevice();
    inputWrapper->copyToDevice();
    outputWrapper->copyToDevice();

    dropoutPropagate1->propagate( batchSize, maskWrapper, inputWrapper, outputWrapper );
    outputWrapper->copyToHost();
    
    int numErrors = 0;
    for( int i = 0; i < outputSize; i++ ) {
        bool ok = true;
        if( ( output[i] > 0 && output0[i] < 0 ) || ( output[i] < 0 && output0[i] > 0 ) ) {
            cout << "signs differ" << endl;
            ok = false;
        }
        if( ok ) {
            if( ( output[i] == 0 && output0[i] != 0 ) || ( output[i] != 0 && output0[i] == 0 ) ) {
                cout << "equality to 0 differs" << endl;
                ok = false;
            }
        }
        if( ok && output[i] != 0 ) {
            if( ( output[i] / output0[i] ) > 1.0001f ) {
                cout << "magnitudes differ 1" << endl;
                ok = false;
            }
            if( ( output0[i] / output[i] ) > 1.0001f ) {
                cout << "magnitudes differ 2" << endl;
                ok = false;
            }
        }
        if( !ok ) {
            cout << "ERROR: output[" << i << "] instance0:" << output0[i] << " != instance1:" << output[i] << endl;
            numErrors++;
        }
        if( numErrors >= 10 ) {
            cout << "More than 10 errors. Skipping the rest :-)" << endl;
            break;
        }
    }
    EXPECT_EQ( 0, numErrors );
    if( numErrors > 0 ) {
        int num2dPlanes = inputSize / imageSize / imageSize;
        for( int plane = 0; plane < num2dPlanes; plane++ ) {
            cout << "2dplane " << plane << ":" << endl;
            for( int i = 0; i < imageSize; i++ ) {
                string line = "";
                for( int j = 0; j < imageSize; j++ ) {
                    line += toString( input[ plane * imageSize * imageSize + i * imageSize + j] ) + " ";
                }
                cout << line << endl;
            }
            cout << endl;
        }
    }

    delete maskWrapper;
    delete inputWrapper;
    delete outputWrapper;
    delete dropoutPropagate0;
    delete dropoutPropagate1;
    delete[] output0;
    delete[] output;
    delete[] input;
    delete[] mask;
    delete cl;
}

TEST( testdropoutpropagate, comparespecific_0_1_dropout2 ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).dropRatio(0.6f)
        .instance0(0).instance1(1) );
}

TEST( testdropoutpropagate, comparespecific_0_1_dropout3 ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).dropRatio(0.6f)
        .instance0(0).instance1(1) );
}

TEST( testdropoutpropagate, comparespecific_0_1_dropout2_pz ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).dropRatio(0.6f)
        .instance0(0).instance1(1) );
}

TEST( testdropoutpropagate, comparespecific_0_1_dropout3_pz ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).dropRatio(0.6f)
        .instance0(0).instance1(1) );
}

TEST( testdropoutpropagate, comparespecific_0_1_dropout3_small ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(1).numPlanes(1).imageSize(2).dropRatio(0.6f)
        .instance0(0).instance1(1) );
}

TEST( testdropoutpropagate, comparespecific_0_1_dropout3_small2 ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(2).numPlanes(1).imageSize(2).dropRatio(0.6f)
        .instance0(0).instance1(1) );
}

TEST( testdropoutpropagate, comparespecific_0_1_dropout3_small2_tanh ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(2).numPlanes(1).imageSize(2).dropRatio(0.6f)
        .instance0(0).instance1(1) );
}

}

