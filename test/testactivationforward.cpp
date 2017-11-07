// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "activate/ActivationForward.h"
#include "activate/ActivationFunction.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/WeightRandomizer.h"

using namespace std;

namespace testactivationforward {

TEST( testactivationforward, basic ) {
    int batchSize = 1;
    int numPlanes = 1;
    int imageSize = 4;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ActivationForward *activationForward = ActivationForward::instanceForTest( cl, numPlanes, imageSize, new ReluActivation() );
    float data[] = { 1, 2, 5, 3,
                     3, 8, 4, 1,
                     3, 33, 14,23,
                     -1, -3.5f,37.4f,5
    };
    int outputNumElements = activationForward->getOutputNumElements( batchSize );
    EXPECT_EQ( outputNumElements, imageSize * imageSize );
    float *output = new float[outputNumElements];

    activationForward->forward( batchSize, data, output );

    EXPECT_EQ( 1, output[0] );
    EXPECT_EQ( 2, output[1] );
    EXPECT_EQ( 5, output[2] );
    EXPECT_EQ( 0, output[12] );
    EXPECT_EQ( 0, output[13] );
    EXPECT_EQ( 37.4f, output[14] );
    EXPECT_EQ( 5, output[15] );

    delete activationForward;
    delete[] output;
    delete cl;
}

TEST( testactivationforward, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int imageSize = 2;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ActivationForward *activationForward = ActivationForward::instanceForTest( cl, numPlanes, imageSize, new ReluActivation() );
    float data[] = { 1, 2, 
                    5, 3,

                     3, 8, 
                    4, 1,

                     3, 33, 
                    14,23,

                     -1, -3.5f,
                    37.4f,5
    };
    int outputNumElements = activationForward->getOutputNumElements( batchSize );
    float *output = new float[outputNumElements];

    activationForward->forward( batchSize, data, output );

    EXPECT_EQ( output[0], 1 );
    EXPECT_EQ( output[1], 2 );
    EXPECT_EQ( output[2], 5 );
    EXPECT_EQ( output[12], 0 );
    EXPECT_EQ( output[13], 0 );
    EXPECT_EQ( output[14], 37.4f );
    EXPECT_EQ( output[15], 5 );

    delete activationForward;
    delete[] output;
    delete cl;
}

TEST( testactivationforward, fromwrappers ) {
    int batchSize = 1;
    int numPlanes = 1;
    int imageSize = 4;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ActivationForward *activationForward = ActivationForward::instanceSpecific( 1, cl, numPlanes, imageSize, new ReluActivation() );
    float input[] = { 1, -2, -5, 3,
                     3, 8, 4, 1,
                     3, 33, 14,23,
                     -1, -3.5f,37.4f,5
    };
    int outputNumElements = activationForward->getOutputNumElements( batchSize );
    float *output = new float[outputNumElements];

    const int inputNumElements = batchSize * numPlanes * imageSize * imageSize;
    CLWrapper *inputWrapper = cl->wrap( inputNumElements, input );
    CLWrapper *outputWrapper = cl->wrap( outputNumElements, output );

    inputWrapper->copyToDevice();

    activationForward->forward( batchSize, inputWrapper, outputWrapper );

    outputWrapper->copyToHost();

    EXPECT_EQ( output[0], 1 );
    EXPECT_EQ( output[1], 0 );
    EXPECT_EQ( output[2], 0 );
    EXPECT_EQ( output[3], 3 );
    EXPECT_EQ( output[12], 0 );
    EXPECT_EQ( output[13], 0 );
    EXPECT_EQ( output[14], 37.4f );
    EXPECT_EQ( output[15], 5 );

    delete inputWrapper;
    delete outputWrapper;
    delete activationForward;
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
    // ints = ['batchSize', 'numPlanes', 'imageSize', 'activationSize', 'instance0', 'instance1' ]
    // strings = ['activation']
    // import cog_fluent
    // cog_fluent.gov3( 'CompareSpecificArgs', ints = ints, floats = floats, strings=strings )
    // ]]]
    // generated, using cog:
    int _batchSize;
    int _numPlanes;
    int _imageSize;
    int _activationSize;
    int _instance0;
    int _instance1;
    std::string _activation;
    CompareSpecificArgs() {
        _batchSize = 0;
        _numPlanes = 0;
        _imageSize = 0;
        _activationSize = 0;
        _instance0 = 0;
        _instance1 = 0;
        _activation = "";
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
    CompareSpecificArgs activationSize( int _activationSize ) {
        this->_activationSize = _activationSize;
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
    CompareSpecificArgs activation( std::string _activation ) {
        this->_activation = _activation;
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

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    ActivationForward *activationForward0 = ActivationForward::instanceSpecific( args._instance0, cl, numPlanes, imageSize, ActivationFunction::fromName( args._activation ) );
    ActivationForward *activationForward1 = ActivationForward::instanceSpecific( args._instance1, cl, numPlanes, imageSize, ActivationFunction::fromName( args._activation ) );

    const int inputNumElements = batchSize * numPlanes * imageSize * imageSize;
    int outputNumElements = activationForward0->getOutputNumElements( batchSize );

    float *input = new float[ inputNumElements ];
    float *output = new float[ outputNumElements ];

    CLWrapper *inputWrapper = cl->wrap( inputNumElements, input );
    CLWrapper *outputWrapper = cl->wrap( outputNumElements, output );

    WeightRandomizer::randomize( input, inputNumElements, -0.1f, 0.1f );

    memset( output, 99, sizeof(int) * outputNumElements );

    inputWrapper->copyToDevice();
    outputWrapper->copyToDevice();

    activationForward0->forward( batchSize, inputWrapper, outputWrapper );
    outputWrapper->copyToHost();

    float *output0 = new float[ outputNumElements ];
    memcpy( output0, output, sizeof(float) * outputNumElements );
    
    memset( output, 99, sizeof(int) * outputNumElements );

    inputWrapper->copyToDevice();
    outputWrapper->copyToDevice();

    activationForward1->forward( batchSize, inputWrapper, outputWrapper );
    outputWrapper->copyToHost();
    
    int numErrors = 0;
    for( int i = 0; i < outputNumElements; i++ ) {
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
    delete outputWrapper;
    delete activationForward0;
    delete activationForward1;
    delete[] output0;
    delete[] output;
    delete[] input;
    delete cl;
}

TEST( testactivationforward, comparespecific_0_1_activation2 ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).activation("relu")
        .instance0(0).instance1(1) );
}

TEST( testactivationforward, comparespecific_0_1_activation3 ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).activation("relu")
        .instance0(0).instance1(1) );
}

TEST( testactivationforward, comparespecific_0_1_activation2_pz ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).activation("relu")
        .instance0(0).instance1(1) );
}

TEST( testactivationforward, comparespecific_0_1_activation3_pz ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(10).numPlanes(5).imageSize(10).activation("relu")
        .instance0(0).instance1(1) );
}

TEST( testactivationforward, comparespecific_0_1_activation3_small ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(1).numPlanes(1).imageSize(2).activation("relu")
        .instance0(0).instance1(1) );
}

TEST( testactivationforward, comparespecific_0_1_activation3_small2 ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(2).numPlanes(1).imageSize(2).activation("relu")
        .instance0(0).instance1(1) );
}

TEST( testactivationforward, comparespecific_0_1_activation3_small2_tanh ) {
    compareSpecific( CompareSpecificArgs::instance()
        .batchSize(2).numPlanes(1).imageSize(2).activation("tanh")
        .instance0(0).instance1(1) );
}

}

