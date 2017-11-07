// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "pooling/PoolingForward.h"
#include "util/stringhelper.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/WeightRandomizer.h"

using namespace std;

namespace testpoolingforward {

TEST( testpoolingforward, basic ) {
    int batchSize = 1;
    int numPlanes = 1;
    int imageSize = 4;
    int poolingSize = 2;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    PoolingForward *poolingForward = PoolingForward::instanceForTest( cl, false, numPlanes, imageSize, poolingSize );
    float data[] = { 1, 2, 5, 3,
                     3, 8, 4, 1,
                     3, 33, 14,23,
                     -1, -3.5f,37.4f,5
    };
    int outputNumElements = poolingForward->getOutputNumElements( batchSize );
    int *selectors = new int[outputNumElements];
    float *output = new float[outputNumElements];

    poolingForward->forward( batchSize, data, selectors, output );

    EXPECT_EQ( selectors[0], 3 );
    EXPECT_EQ( selectors[1], 0 );
    EXPECT_EQ( selectors[2], 1 );
    EXPECT_EQ( selectors[3], 2 );

    EXPECT_EQ( output[0], 8 );
    EXPECT_EQ( output[1], 5 );
    EXPECT_EQ( output[2], 33 );
    EXPECT_EQ( output[3], 37.4f );

    delete poolingForward;
    delete[] selectors;
    delete[] output;
    delete cl;
}

TEST( testpoolingforward, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int imageSize = 2;
    int poolingSize = 2;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    PoolingForward *poolingForward = PoolingForward::instanceForTest( cl, false, numPlanes, imageSize, poolingSize );
    float data[] = { 1, 2, 
                    5, 3,

                     3, 8, 
                    4, 1,

                     3, 33, 
                    14,23,

                     -1, -3.5f,
                    37.4f,5
    };
    int outputNumElements = poolingForward->getOutputNumElements( batchSize );
    int *selectors = new int[outputNumElements];
    float *output = new float[outputNumElements];

    poolingForward->forward( batchSize, data, selectors, output );

    EXPECT_EQ( selectors[0], 2 );
    EXPECT_EQ( selectors[1], 1 );
    EXPECT_EQ( selectors[2], 1 );
    EXPECT_EQ( selectors[3], 2 );

    EXPECT_EQ( output[0], 5 );
    EXPECT_EQ( output[1], 8 );
    EXPECT_EQ( output[2], 33 );
    EXPECT_EQ( output[3], 37.4f );

    delete poolingForward;
    delete[] selectors;
    delete[] output;
    delete cl;
}

TEST( testpoolingforward, fromwrappers ) {
    int batchSize = 1;
    int numPlanes = 1;
    int imageSize = 4;
    int poolingSize = 2;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    PoolingForward *poolingForward = PoolingForward::instanceSpecific( 1, cl, false, numPlanes, imageSize, poolingSize );
    float input[] = { 1, 2, 5, 3,
                     3, 8, 4, 1,
                     3, 33, 14,23,
                     -1, -3.5f,37.4f,5
    };
    int outputNumElements = poolingForward->getOutputNumElements( batchSize );
    int *selectors = new int[outputNumElements];
    float *output = new float[outputNumElements];

    const int inputNumElements = batchSize * numPlanes * imageSize * imageSize;
    CLWrapper *inputWrapper = cl->wrap( inputNumElements, input );
    CLWrapper *selectorsWrapper = cl->wrap( outputNumElements, selectors );
    CLWrapper *outputWrapper = cl->wrap( outputNumElements, output );

    inputWrapper->copyToDevice();

    poolingForward->forward( batchSize, inputWrapper, selectorsWrapper, outputWrapper );

    selectorsWrapper->copyToHost();
    outputWrapper->copyToHost();

    EXPECT_EQ( selectors[0], 3 );
    EXPECT_EQ( selectors[1], 0 );
    EXPECT_EQ( selectors[2], 1 );
    EXPECT_EQ( selectors[3], 2 );

    EXPECT_EQ( output[0], 8 );
    EXPECT_EQ( output[1], 5 );
    EXPECT_EQ( output[2], 33 );
    EXPECT_EQ( output[3], 37.4f );

    delete inputWrapper;
    delete selectorsWrapper;
    delete outputWrapper;
    delete poolingForward;
    delete[] selectors;
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
    // floats= []
    // ints = ['batchSize', 'numPlanes', 'imageSize', 'poolingSize', 'instance0', 'instance1', 'padZeros' ]
    // import cog_fluent
    // cog_fluent.gov3( 'CompareSpecificArgs', ints = ints, floats = floats )
    // ]]]
    // generated, using cog:
    int _batchSize;
    int _numPlanes;
    int _imageSize;
    int _poolingSize;
    int _instance0;
    int _instance1;
    int _padZeros;
    CompareSpecificArgs() {
        _batchSize = 0;
        _numPlanes = 0;
        _imageSize = 0;
        _poolingSize = 0;
        _instance0 = 0;
        _instance1 = 0;
        _padZeros = 0;
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
    CompareSpecificArgs poolingSize( int _poolingSize ) {
        this->_poolingSize = _poolingSize;
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
    CompareSpecificArgs padZeros( int _padZeros ) {
        this->_padZeros = _padZeros;
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
    int poolingSize = args._poolingSize;

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    PoolingForward *poolingForward0 = PoolingForward::instanceSpecific( args._instance0, cl, args._padZeros, numPlanes, imageSize, poolingSize );
    PoolingForward *poolingForward1 = PoolingForward::instanceSpecific( args._instance1, cl, args._padZeros, numPlanes, imageSize, poolingSize );

    const int inputNumElements = batchSize * numPlanes * imageSize * imageSize;
    int outputNumElements = poolingForward0->getOutputNumElements( batchSize );

    float *input = new float[ inputNumElements ];
    int *selectors = new int[ outputNumElements ];
    float *output = new float[ outputNumElements ];

    CLWrapper *inputWrapper = cl->wrap( inputNumElements, input );
    CLWrapper *selectorsWrapper = cl->wrap( outputNumElements, selectors );
    CLWrapper *outputWrapper = cl->wrap( outputNumElements, output );

    WeightRandomizer::randomize( input, inputNumElements, -0.1f, 0.1f );

    memset( selectors, 99, sizeof(int) * outputNumElements );
    memset( output, 99, sizeof(int) * outputNumElements );

    inputWrapper->copyToDevice();
    selectorsWrapper->copyToDevice();
    outputWrapper->copyToDevice();

    poolingForward0->forward( batchSize, inputWrapper, selectorsWrapper, outputWrapper );
    selectorsWrapper->copyToHost();
    outputWrapper->copyToHost();

    int *selectors0 = new int[ outputNumElements ];
    float *output0 = new float[ outputNumElements ];
    memcpy( selectors0, selectors, sizeof(int) * outputNumElements );
    memcpy( output0, output, sizeof(float) * outputNumElements );
    
    memset( selectors, 99, sizeof(int) * outputNumElements );
    memset( output, 99, sizeof(int) * outputNumElements );

    inputWrapper->copyToDevice();
    selectorsWrapper->copyToDevice();
    outputWrapper->copyToDevice();

    poolingForward1->forward( batchSize, inputWrapper, selectorsWrapper, outputWrapper );
    selectorsWrapper->copyToHost();
    outputWrapper->copyToHost();
    
    int numErrors = 0;
    for( int i = 0; i < outputNumElements; i++ ) {
        if( selectors[i] != selectors0[i] ) {
            cout << "ERROR: selectors[" << i << "] instance0:" << selectors0[i] << " != instance1:" << selectors[i] << endl;
            numErrors++;
        }
        if( output[i] != output0[i] ) {
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
        int num2dPlanes = inputNumElements / imageSize / imageSize;
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

    delete inputWrapper;
    delete selectorsWrapper;
    delete outputWrapper;
    delete poolingForward0;
    delete poolingForward1;
    delete[] selectors0;
    delete[] output0;
    delete[] selectors;
    delete[] output;
    delete[] input;
    delete cl;
}

TEST( testpoolingforward, comparespecific_0_1_pooling2 ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).poolingSize(2)
        .instance0(0).instance1(1) );
}

TEST( testpoolingforward, comparespecific_0_1_pooling3 ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).poolingSize(3)
        .instance0(0).instance1(1) );
}

TEST( testpoolingforward, comparespecific_0_1_pooling2_pz ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).poolingSize(2)
        .instance0(0).instance1(1).padZeros(1) );
}

TEST( testpoolingforward, comparespecific_0_1_pooling3_pz ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).poolingSize(3)
        .instance0(0).instance1(1).padZeros(1) );
}

TEST( testpoolingforward, comparespecific_0_1_pooling3_small ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(1).numPlanes(1).imageSize(2).poolingSize(3)
        .instance0(0).instance1(1).padZeros(1) );
}

TEST( testpoolingforward, comparespecific_0_1_pooling3_small2 ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(2).numPlanes(1).imageSize(2).poolingSize(3)
        .instance0(0).instance1(1).padZeros(1) );
}

}

