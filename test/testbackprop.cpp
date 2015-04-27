// this will check the backprop process, and regression test it against original output, with
// original slow procedure

#include <iostream>
#include <random>

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/Sampler.h"

#include "ConvolutionalLayer.h"
#include "NeuralNet.h"

using namespace std;

void printSamples(int weightsSize, float *weightChanges, int numSamples = 5 ) {
    mt19937 random;
    random.seed(0);
    for( int sample = 0; sample < numSamples; sample++ ) {
        int index = random() % weightsSize;
        cout << "EXPECT_FLOAT_NEAR( " << weightChanges[index] << ", weightChanges[" << index << "] );" << endl;
    }
}

// this will test layer 1 backprop in a network like:
//    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(28)->instance();
//    net->convolutionalMaker()->numFilters(14)->filterSize(5)->tanh()->biased()->insert();
//    net->convolutionalMaker()->numFilters(10)->filterSize(24)->tanh()->biased(config.biased)->insert();
/*TEST( testbackprop, main ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(28)->instance();
    net->convolutionalMaker()->numFilters(14)->filterSize(5)->tanh()->biased()->insert();
    net->convolutionalMaker()->numFilters(10)->filterSize(24)->tanh()->biased()->insert();
    net->setBatchSize(128);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    CLWrapper *errorsWrapper = net->cl->wrap( outputSize, errors );
    errorsWrapper->copyToDevice();
    layer1->outputWrapper->copyToDevice();
    CLWrapper *imagesWrapper = net->cl->wrap( net->layers[0]->getOutputSize(), net->layers[0]->getOutput() );
    imagesWrapper->copyToDevice();

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        memset( layer1->weights, 0, sizeof(float) * weightsSize );
        layer1->weightsWrapper->copyToDevice();

        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, imagesWrapper, layer1->outputWrapper, errorsWrapper, layer1->weightsWrapper );
//        layer1->backwardWeightsCpu( 0.1f, errors, layer1->weights );
//        cout << "after backprop" << endl;
        StatefulTimer::timeCheck("after backprop");
        layer1->weightsWrapper->copyToHost();

        Sampler::printSamples( "layer1->weights", weightsSize, layer1->weights );

        EXPECT_FLOAT_NEAR(-3.28283e-05,  layer1->weights[33] );
        EXPECT_FLOAT_NEAR( -1.15271e-05,  layer1->weights[144] );
        EXPECT_FLOAT_NEAR( -9.22672e-06,  layer1->weights[339] );
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete errorsWrapper;
    delete imagesWrapper;
//    delete[]weightChanges;
//    delete[]weights;
    delete[] upstreamOutput;
}
 */   
// this will test layer 1 backprop in a network like:
//    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(19)->instance();
//    net->convolutionalMaker()->numFilters(32)->filterSize(5)->tanh()->biased()->insert();
//    net->convolutionalMaker()->numFilters(10)->filterSize(15)->tanh()->biased(config.biased)->insert();
/*TEST( testbackprop, image19 ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(19)->instance();
    net->convolutionalMaker()->numFilters(32)->filterSize(5)->tanh()->biased()->insert();
//    net->convolutionalMaker()->numFilters(10)->filterSize(15)->tanh()->biased()->insert();
    net->setBatchSize(128);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    CLWrapper *weightsWrapper = net->cl->wrap( weightsSize, weights );
    memset( weights, 0, sizeof(float) * weightsSize );
    weightsWrapper->copyToDevice();

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    CLWrapper *errorsWrapper = net->cl->wrap( outputSize, errors );
    errorsWrapper->copyToDevice();
    layer1->outputWrapper->copyToDevice();
    CLWrapper *imagesWrapper = net->cl->wrap( net->layers[0]->getOutputSize(), net->layers[0]->getOutput() );
    imagesWrapper->copyToDevice();

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 3; i++ ) {
        memset( weights, 0, sizeof(float) * weightsSize );
        weightsWrapper->copyToDevice();

        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, imagesWrapper, layer1->outputWrapper, errorsWrapper, weightsWrapper );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;
        StatefulTimer::timeCheck("after backprop");

//        printSamples( weightsSize, weightChanges );
        weightsWrapper->copyToHost();
        EXPECT_FLOAT_NEAR( -1.35891e-05, weights[44] );
        EXPECT_FLOAT_NEAR( 1.86079e-05, weights[239] );
        EXPECT_FLOAT_NEAR( -8.72368e-06, weights[533] );
        EXPECT_FLOAT_NEAR( -3.20888e-05, weights[160] );
        EXPECT_FLOAT_NEAR( 9.92044e-06, weights[163] );

    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete errorsWrapper;
    delete[] errors;
//    delete[]weightChanges;
//    delete[]weights;
    delete[] upstreamOutput;
}
*/
/*
TEST( testbackprop, image19_1plane_1filter ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(19)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(5)->tanh()->biased()->insert();
//    net->convolutionalMaker()->numFilters(10)->filterSize(15)->tanh()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    memset( weights, 0, sizeof(float) * weightsSize );
    CLWrapper *weightsWrapper = net->cl->wrap( weightsSize, weights );
    weightsWrapper->copyToDevice();

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    CLWrapper *errorsWrapper = net->cl->wrap( outputSize, errors );
    errorsWrapper->copyToDevice();
    layer1->outputWrapper->copyToDevice();
    CLWrapper *imagesWrapper = net->cl->wrap( net->layers[0]->getOutputSize(), net->layers[0]->getOutput() );
    imagesWrapper->copyToDevice();

    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, imagesWrapper, layer1->outputWrapper, errorsWrapper, weightsWrapper );
        StatefulTimer::timeCheck("after backprop");

        weightsWrapper->copyToHost();
        for( int i = 0; i < 15; i++ ) {
            cout << "weightChanges[" << i << "]=" << weights[i] << endl;
        }

        printSamples( weightsSize, weights );

        EXPECT_FLOAT_NEAR( -4.62887e-05, weights[19] );
        EXPECT_FLOAT_NEAR( 0.000642478, weights[14] );
        EXPECT_FLOAT_NEAR( 0.000225103, weights[8] );
        EXPECT_FLOAT_NEAR( 0.000378945, weights[10] );
        EXPECT_FLOAT_NEAR( 2.41975e-05, weights[13] );

    }
    StatefulTimer::dump(true);

    delete[] errors;
    delete errorsWrapper;
//    delete[]weightChanges;
//    delete[]weights;
    delete[] upstreamOutput;
}
*/
/*
TEST( testbackprop, image19_1plane_1filter_batchsize128 ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(19)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(5)->tanh()->biased()->insert();
//    net->convolutionalMaker()->numFilters(10)->filterSize(15)->tanh()->biased()->insert();
    net->setBatchSize(128);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
        StatefulTimer::timeCheck("after backprop");

        printSamples( weightsSize, weightChanges );

        EXPECT_FLOAT_NEAR( 3.75026e-05, weightChanges[19] );
        EXPECT_FLOAT_NEAR( -1.69212e-06, weightChanges[14] );
        EXPECT_FLOAT_NEAR( -1.91246e-05, weightChanges[8] );
        EXPECT_FLOAT_NEAR( -7.15637e-05, weightChanges[10] );
        EXPECT_FLOAT_NEAR( 1.63161e-05, weightChanges[13] );


    }
    StatefulTimer::dump(true);

    delete[] errors;
    delete[]weightChanges;
//    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image19_1plane_2filter_batchsize128 ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(19)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(5)->tanh()->biased()->insert();
//    net->convolutionalMaker()->numFilters(10)->filterSize(15)->tanh()->biased()->insert();
    net->setBatchSize(128);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
        StatefulTimer::timeCheck("after backprop");

        printSamples( weightsSize, weightChanges );

        EXPECT_FLOAT_NEAR( 3.31051e-05, weightChanges[44] );
        EXPECT_FLOAT_NEAR( 1.59785e-05, weightChanges[39] );
        EXPECT_FLOAT_NEAR( 9.45172e-06, weightChanges[33] );
        EXPECT_FLOAT_NEAR( 4.63717e-06, weightChanges[10] );
        EXPECT_FLOAT_NEAR( 2.23327e-05, weightChanges[13] );



    }
    StatefulTimer::dump(true);

    delete[] errors;
    delete[]weightChanges;
//    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize5 ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(5)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(3)->relu()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;
        StatefulTimer::timeCheck("after backprop");
        random.seed(0);
        for( int sample = 0; sample < 3; sample++ ) {
            int index = random() % weightsSize;
            cout << "weightChanges[" << index << "]=" << weightChanges[index] << endl;
        }
//        if( false ) {
            EXPECT_FLOAT_NEAR(-0.000153949,  weightChanges[8] );
            EXPECT_FLOAT_NEAR( -0.000229652,  weightChanges[0] );
            EXPECT_FLOAT_NEAR( -0.000262128,  weightChanges[2] );
//        }
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize5_tanh ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(5)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(3)->tanh()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;
        StatefulTimer::timeCheck("after backprop");
        random.seed(0);
        for( int sample = 0; sample < 3; sample++ ) {
            int index = random() % weightsSize;
            cout << "weightChanges[" << index << "]=" << weightChanges[index] << endl;
        }
//        if( false ) {
            EXPECT_FLOAT_NEAR(6.65251e-05,  weightChanges[8] );
            EXPECT_FLOAT_NEAR( -0.000491319,  weightChanges[0] );
            EXPECT_FLOAT_NEAR( -0.000609587,  weightChanges[2] );
//        }
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize7_filtersize5 ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(7)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(5)->relu()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;
        StatefulTimer::timeCheck("after backprop");
        random.seed(0);
        for( int sample = 0; sample < 3; sample++ ) {
            int index = random() % weightsSize;
            cout << "weightChanges[" << index << "]=" << weightChanges[index] << endl;
        }
//        if( false ) {
            EXPECT_FLOAT_NEAR(-0.000242163,  weightChanges[19] );
            EXPECT_FLOAT_NEAR( -0.000223168,  weightChanges[14] );
            EXPECT_FLOAT_NEAR( 7.97735e-05,  weightChanges[8] );
//        }
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize11_filtersize5_tanh ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(11)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(5)->tanh()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;
        StatefulTimer::timeCheck("after backprop");
        random.seed(0);
        for( int sample = 0; sample < 3; sample++ ) {
            int index = random() % weightsSize;
            cout << "weightChanges[" << index << "]=" << weightChanges[index] << endl;
        }
//        if( false ) {
            EXPECT_FLOAT_NEAR(5.64488e-05,  weightChanges[19] );
            EXPECT_FLOAT_NEAR( -0.000566337,  weightChanges[14] );
            EXPECT_FLOAT_NEAR( -7.05525e-05,  weightChanges[8] );
//        }
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize15_filtersize5_tanh ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(15)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(5)->tanh()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;
        StatefulTimer::timeCheck("after backprop");
        random.seed(0);
        for( int sample = 0; sample < 3; sample++ ) {
            int index = random() % weightsSize;
            cout << "EXPECT_FLOAT_NEAR( " << weightChanges[index] << ", weightChanges[" << index << "] );" << endl;
        }
//        if( false ) {
        EXPECT_FLOAT_NEAR( 0.000320274, weightChanges[19] );
        EXPECT_FLOAT_NEAR( 0.000764995, weightChanges[14] );
        EXPECT_FLOAT_NEAR( 0.000302961, weightChanges[8] );

//        }
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize16_filtersize5_tanh ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(16)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(5)->tanh()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;

        printSamples( weightsSize, weightChanges );

        StatefulTimer::timeCheck("after backprop");
        EXPECT_FLOAT_NEAR( -0.000729529, weightChanges[19] );
        EXPECT_FLOAT_NEAR( -0.000510299, weightChanges[14] );
        EXPECT_FLOAT_NEAR( 8.423e-05, weightChanges[8] );
        EXPECT_FLOAT_NEAR( -0.000237997, weightChanges[10] );
        EXPECT_FLOAT_NEAR( -0.000174111, weightChanges[13] );

    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize17_filtersize5_tanh ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(17)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(5)->tanh()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;

        printSamples( weightsSize, weightChanges );

        StatefulTimer::timeCheck("after backprop");
        EXPECT_FLOAT_NEAR( -0.000565815, weightChanges[19] );
        EXPECT_FLOAT_NEAR( 4.88314e-05, weightChanges[14] );
        EXPECT_FLOAT_NEAR( 0.000305842, weightChanges[8] );
        EXPECT_FLOAT_NEAR( 0.000279056, weightChanges[10] );
        EXPECT_FLOAT_NEAR( -0.000279087, weightChanges[13] );

        for ( int i = 0; i < 20; i++ ) {
            cout << " weightChanges[" << i << "]=" << weightChanges[i] << endl;
        }
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize17_filtersize3_tanh ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(17)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(3)->tanh()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;

        printSamples( weightsSize, weightChanges );

        StatefulTimer::timeCheck("after backprop");
        EXPECT_FLOAT_NEAR( 3.38554e-06, weightChanges[8] );
        EXPECT_FLOAT_NEAR( -0.000101404, weightChanges[0] );
        EXPECT_FLOAT_NEAR( -7.10894e-05, weightChanges[2] );
        EXPECT_FLOAT_NEAR( -0.00016516, weightChanges[6] );
        EXPECT_FLOAT_NEAR( 0.00050475, weightChanges[7] );

        for ( int i = 0; i < 20; i++ ) {
            cout << " weightChanges[" << i << "]=" << weightChanges[i] << endl;
        }
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize17_filtersize1_tanh ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(17)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(1)->tanh()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;

        printSamples( weightsSize, weightChanges );

        StatefulTimer::timeCheck("after backprop");
        EXPECT_FLOAT_NEAR( -0.000217034, weightChanges[0] );

        for ( int i = 0; i < 20; i++ ) {
            cout << " weightChanges[" << i << "]=" << weightChanges[i] << endl;
        }
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize17_filtersize1_linear ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(17)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(1)->linear()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;

        printSamples( weightsSize, weightChanges );

        StatefulTimer::timeCheck("after backprop");
        EXPECT_FLOAT_NEAR(  -0.000218822, weightChanges[0] );

        for ( int i = 0; i < 20; i++ ) {
            cout << " weightChanges[" << i << "]=" << weightChanges[i] << endl;
        }
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize17_filtersize1_linear_diffvalues ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(17)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(1)->linear()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 2.0f - 1.0f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();

//    float *weights = new float[weightsSize];
//    for( int i = 0; i < weightsSize; i++ ) {
//        weights[i] = random() / (float)mt19937::max() * 2.0f - 1.0f;
//    }
//    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    float guessValue = 0;
    float guessValue2 = 0;
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 2.0f - 1.0f;
        layer1->output[i] = random() / (float)mt19937::max() * 2.0f - 1.0f;
        guessValue += errors[i] * upstreamOutput[i];
        if( i < 256 ) {
            guessValue2 += errors[i] * upstreamOutput[i];
        }
    }
    cout << " guessvalue: " << guessValue * ( - 0.00588235 ) << " guessvalue2: " << guessValue2 * ( - 0.00588235 ) << endl;

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;

        printSamples( weightsSize, weightChanges, 1 );

        StatefulTimer::timeCheck("after backprop");
        EXPECT_FLOAT_NEAR(  0.0564353, weightChanges[0] );

        for ( int i = 0; i < 15; i++ ) {
            cout << " weightChanges[" << i << "]=" << weightChanges[i] << endl;
        }
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
//    delete[]weights;
    delete[] upstreamOutput;
}

TEST( testbackprop, image5_1plane_1filter_upstreamimagesize19_filtersize5_tanh ) {
    mt19937 random;
    random.seed(0); // so always gives same output
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(19)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(5)->tanh()->biased()->insert();
    net->setBatchSize(1);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 

    int upstreamOutputSize = net->layers[0]->getOutputSize();
    float *upstreamOutput = new float[upstreamOutputSize];
    for( int i = 0; i < upstreamOutputSize; i++ ) {
        upstreamOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamOutput);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int outputSize = layer1->getOutputSize();
    float *errors = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamoutputsize " << upstreamOutputSize << " outputSize " << outputSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 1; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backwardWeightsGpuWithScratch( 0.1f, errors, weightChanges );
//        layer1->backwardWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;

        printSamples( weightsSize, weightChanges );

        StatefulTimer::timeCheck("after backprop");
        EXPECT_FLOAT_NEAR( -8.80638e-06, weightChanges[19] );
        EXPECT_FLOAT_NEAR( -1.43798e-05, weightChanges[14] );
        EXPECT_FLOAT_NEAR( 0.000190677, weightChanges[8] );
        EXPECT_FLOAT_NEAR( -4.81357e-05, weightChanges[10] );
        EXPECT_FLOAT_NEAR( 0.000189767, weightChanges[13] );
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamOutput;
}

*/

