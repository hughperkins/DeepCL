// tests the time for a single batch to go forward and backward through the layers

#include <iostream>
#include <random>
#include <cstring>
#include <cmath>

#include "net/NeuralNet.h"
#include "weights/WeightsPersister.h"
#include "layer/Layer.h"
#include "conv/ConvolutionalLayer.h"
#include "net/NeuralNetMould.h"
#include "layer/LayerMakers.h"
#include "trainers/SGD.h"
#include "clblas/ClBlasInstance.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/Sampler.h"
#include "test/TestArgsParser.h"
#include "util/Timer.h"

//#if (_MSC_VER == 1500 || _MSC_VER == 1600)
#ifdef _MSC_VER // make consistent across all msvc versions, so dont have to retest on different msvc versions...
#define TR1RANDOM
typedef std::tr1::mt19937 MT19937;
#define isnan _isnan
#else
typedef std::mt19937 MT19937;
#endif

using namespace std;

class TestArgs{
public:
    static TestArgs instance(){ TestArgs args; return args; };
    // [[[cog
    // floats= ['learningRate']
    // ints = ['batchSize','imageSize','numLayers','filterSize','numFilters', 'numEpochs', 'poolingSize',
    // 'numCats', 'softMax' ]
    // import cog_fluent
    // cog_fluent.go1b('TestArgs', ints = ints, floats = floats)
    // ]]]
    // generated, using cog:
    int batchSize;
    int imageSize;
    int numLayers;
    int filterSize;
    int numFilters;
    int numEpochs;
    int poolingSize;
    int numCats;
    int softMax;
    float learningRate;
    TestArgs() {
        batchSize = 0;
        imageSize = 0;
        numLayers = 0;
        filterSize = 0;
        numFilters = 0;
        numEpochs = 0;
        poolingSize = 0;
        numCats = 0;
        softMax = 0;
        learningRate = 0;
    }
    TestArgs BatchSize( int _batchSize ) {
        this->batchSize = _batchSize;
        return *this;
    }
    TestArgs ImageSize( int _imageSize ) {
        this->imageSize = _imageSize;
        return *this;
    }
    TestArgs NumLayers( int _numLayers ) {
        this->numLayers = _numLayers;
        return *this;
    }
    TestArgs FilterSize( int _filterSize ) {
        this->filterSize = _filterSize;
        return *this;
    }
    TestArgs NumFilters( int _numFilters ) {
        this->numFilters = _numFilters;
        return *this;
    }
    TestArgs NumEpochs( int _numEpochs ) {
        this->numEpochs = _numEpochs;
        return *this;
    }
    TestArgs PoolingSize( int _poolingSize ) {
        this->poolingSize = _poolingSize;
        return *this;
    }
    TestArgs NumCats( int _numCats ) {
        this->numCats = _numCats;
        return *this;
    }
    TestArgs SoftMax( int _softMax ) {
        this->softMax = _softMax;
        return *this;
    }
    TestArgs LearningRate( float _learningRate ) {
        this->learningRate = _learningRate;
        return *this;
    }
    // [[[end]]]
};

// this tests whether the change in loss predicted by the change in weights
// is approximately correct
// the change in weights predicts the change in loss, because:
// - the gradient of loss with respect to weights tells us how much
//   the loss should change with respect ot a weights change
// - the size of the weights change tells us the gradient of the loss 
//   with respect to the weights :-)
// therefore, the square of the change in weights should approximately
// equal the change in loss
// that's what this method checks
// whether this is teh best way to do this is open to question
// for alternative approaches, look at:
//    - testbackward.checkLayer(), or
//    - testupdateweights.test()
void test(float learningRate, int numEpochs, int batchSize, NeuralNet *net, float tolerance = 1.3f) {
    net->setBatchSize(batchSize);
    MT19937 random;
    random.seed(1); // so always gives same output
    const int inputsSize = net->getInputCubeSize() * batchSize;
    float *inputData = new float[ inputsSize ];
    for(int i = 0; i < inputsSize; i++) {
        inputData[i] = random() / (float)random.max() * 0.2f - 0.1f;
    }
    const int outputNumElements = net->getLastLayer()->getOutputNumElements();
    float *expectedOutput = new float[outputNumElements];
    for(int i = 0; i < outputNumElements; i++) {
        expectedOutput[i] = random() / (float)random.max() * 0.2f - 0.1f;
    }

    for (int layerIndex = 1; layerIndex < net->getNumLayers(); layerIndex++) {
        ConvolutionalLayer *layer = dynamic_cast< ConvolutionalLayer * >(net->getLayer(layerIndex));
        if(layer == 0) {
            continue;
        }
        int weightsSize = layer->getWeightsSize();
//        cout << "weightsSize, layer " << 1 << "= " << weightsSize << endl;
        for(int i = 0; i < weightsSize; i++) {
            layer->weights[i] = random() / (float)random.max() * 0.2f - 0.1f;
        }
        layer->weightsWrapper->copyToDevice();
        int biasSize = layer->getBiasSize();
        for(int i = 0; i < biasSize; i++) {
            layer->bias[i] = random() / (float)random.max() * 0.2f - 0.1f;
        }
    }

    Timer timer;
    int weightsTotalSize = WeightsPersister::getTotalNumWeights(net);
    cout << "weightsTotalSize=" << weightsTotalSize << endl;
    float *lastWeights = new float[weightsTotalSize];
    float *currentWeights = new float[weightsTotalSize];
    WeightsPersister::copyNetWeightsToArray(net, lastWeights);
//    Sampler::sampleFloats("lastWeights", weightsTotalSize, lastWeights);
//    cout << "learningRate: " << args.learningRate << endl;
    float lastloss = 0;
    bool allOk = true;
    SGD *sgd = SGD::instance(net->getCl(), learningRate, 0.0f);
    for(int i = 0; i < numEpochs; i++) {
//        net->learnBatch(args.learningRate, inputData, expectedOutput);
        TrainingContext context(0, 0);
        sgd->train(net, &context, inputData, expectedOutput);
//        net->forward(inputData);
//        net->backward(expectedOutput);
        WeightsPersister::copyNetWeightsToArray(net, currentWeights);
//        Sampler::sampleFloats("currentWeights", weightsTotalSize, currentWeights);
        float sumsquaredweightsdiff = 0;
        for(int j = 0; j < weightsTotalSize; j++) {
            float thisdiff = currentWeights[j] - lastWeights[j];
            sumsquaredweightsdiff += thisdiff * thisdiff;
        }
        float lossChangeFromW = (sumsquaredweightsdiff/learningRate);
        net->forward(inputData);
        float thisloss = net->calcLoss(expectedOutput) ;
        float lossChange = (lastloss - thisloss);
//        cout << "i=" << i << endl;
////        cout << "loss " << thisloss << " loss diff " << lossChange << endl;
////        cout << "compare:" << endl;
//        cout << "    losschangefromw " << lossChangeFromW << endl;
//        cout << "    actual loss change " << lossChange << endl;
        if(isnan(lossChange)) {
            cout << "DIFF, epoch=" << i << " :" << endl;
            cout << "    losschangefromw " << lossChangeFromW << endl;
            cout << "    actual loss change " << lossChange << endl;
            allOk = false;
//            EXPECT_TRUE(!isnan(lossChange));
        }
        if(lossChange / lossChangeFromW > tolerance) {
            cout << "tolerance " << tolerance << endl;
            cout << "DIFF, epoch=" << i << " :" << endl;
            cout << "    losschangefromw " << lossChangeFromW << endl;
            cout << "    actual loss change " << lossChange << endl;
//            cout << "loss: " << lastloss << " -> " << thisloss << endl;
//            EXPECT_EQ(lossChange, lossChangeFromW);
            allOk = false;
        } else if(lossChangeFromW / lossChange > tolerance) {
            cout << "tolerance " << tolerance << endl;
            cout << "DIFF, epoch=" << i << " :" << endl;
            cout << "    losschangefromw " << lossChangeFromW << endl;
            cout << "    actual loss change " << lossChange << endl;
//            cout << "loss: " << lastloss << " -> " << thisloss << endl;
//            EXPECT_EQ(lossChange, lossChangeFromW);
            allOk = false;
        }
        lastloss =thisloss;
//        memcpy(lastWeights1, currentWeights1, sizeof(float) * weights1Size);
        WeightsPersister::copyNetWeightsToArray(net, lastWeights);
//        Sampler::sampleFloats("lastWeights", weightsTotalSize, lastWeights);
    }
    timer.timeCheck("batch time");
    StatefulTimer::dump(true);

    EXPECT_EQ(true, allOk);

//    float *output = (float*)(net->getOutput());
//    Sampler::printSamples("net->getOutput()", outputNumElements, (float*)output);

    delete sgd;
    delete[]currentWeights;
    delete[]lastWeights;
    delete[] expectedOutput;
    delete[] inputData;
}

void test(ActivationFunction *fn, TestArgs args, float tolerance = 1.3f) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->planes(1)->imageSize(args.imageSize)->instance();
    for(int i = 0; i < args.numLayers; i++) {
        net->addLayer(ConvolutionalMaker::instance()->numFilters(args.numFilters)->filterSize(args.filterSize)->biased());
        net->addLayer(ActivationMaker::instance()->fn(fn));
        if(args.poolingSize > 0) {
            net->addLayer(PoolingMaker::instance()->poolingSize(args.poolingSize));
        }
    }
    if(args.softMax) {
        net->addLayer(FullyConnectedMaker::instance()->numPlanes(args.numFilters)->imageSize(1)->biased());
        net->addLayer(SoftMaxMaker::instance());
    } else {
        net->addLayer(FullyConnectedMaker::instance()->numPlanes(args.numFilters)->imageSize(1)->biased());
        net->addLayer(ActivationMaker::instance()->tanh());
        net->addLayer(SquareLossMaker::instance());;
    }
    net->print();
    test(args.learningRate, args.numEpochs, args.batchSize, net, tolerance);
    delete net;
    delete cl;
}

TEST(testsinglebatch, imagesize5_filtersize3_batchsize2) {
    test(new LinearActivation(), TestArgs::instance().BatchSize(2).LearningRate(0.00001f).ImageSize(5).NumLayers(1)
            .FilterSize(3).NumFilters(5).NumEpochs(20));
}

TEST(EXCLUDED_testsinglebatch, imagesize5_filtersize3_batchsize2_10filters) {
    test(new ReluActivation(), TestArgs::instance().BatchSize(2).LearningRate(0.01f).ImageSize(5).NumLayers(1)
            .FilterSize(3).NumFilters(10).NumEpochs(100));
}

TEST(testsinglebatch, imagesize28) {
    test(new ReluActivation(), TestArgs::instance().BatchSize(2).LearningRate(0.001f).ImageSize(28).NumLayers(1)
            .FilterSize(3).NumFilters(10).NumEpochs(100), 2.5f);
}

TEST(testsinglebatch, imagesize28_filtersize5) {
    test(new ReluActivation(), TestArgs::instance().BatchSize(2).LearningRate(0.001f).ImageSize(28).NumLayers(1)
            .FilterSize(5).NumFilters(10).NumEpochs(100), 1.5f);
}

float sumWeightChangesSquared(float *__restrict oldWeights, float * __restrict newWeights, NeuralNet *net) {
    int offset = 0;
    float sum = 0;
    for(int i = 1; i < (int)net->getNumLayers(); i++) {
        Layer *layer = net->getLayer(i);
        int numWeights = layer->getPersistSize();
        if(numWeights == 0) {
            continue;
        }
        float layerSum = 0;
        for(int i = 0; i < numWeights; i++) {
            float thisDiff = newWeights[offset] - oldWeights[offset];
            layerSum += thisDiff * thisDiff;
            offset++;
        }
        EXPECT_NE(0.0f, layerSum);
        sum += layerSum;
    }
    return sum;
}

void checkErrorsForLayer(int layerId, float lastLoss, NeuralNet *net, float *lastWeights, float *currentWeights, float learningRate, float *inputData, int *labels) {
    // rollback all our changes, except our layer
    WeightsPersister::copyArrayToNetWeights(lastWeights, net);
    int offset = WeightsPersister::getArrayOffsetForLayer(net, layerId);
    cout << "layer " << layerId << " offset: " << offset << endl;
    Layer *layer = net->getLayer(layerId);
    int numWeights = layer->getPersistSize();
    if(numWeights == 0) {
        return;
    }
    layer->unpersistFromArray(currentWeights + offset);
//    layer->unpersistFromArray(lastWeights + offset);
    float thisWSquaredDiffSum = 0;
    for(int i = 0; i < numWeights; i++) {
        float thisDiff = currentWeights[i + offset] - lastWeights[i + offset];
        thisWSquaredDiffSum += thisDiff * thisDiff;
    }
    float lossChangeFromW = thisWSquaredDiffSum / learningRate;
    net->forward(inputData);
    float newLoss = net->calcLossFromLabels(labels);
    cout << "layer " << layerId << endl;
    cout << "    " << "from w: " << lossChangeFromW << endl;
    cout << "    " << "actual: " << (lastLoss - newLoss) << endl;
}

void testLabelled(TestArgs args) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->planes(1)->imageSize(args.imageSize)->instance();
    for(int i = 0; i < args.numLayers; i++) {
        net->addLayer(ConvolutionalMaker::instance()->numFilters(args.numFilters)->filterSize(args.filterSize)->biased()->padZeros());
        net->addLayer(ActivationMaker::instance()->relu());
        if(args.poolingSize > 0) {
            net->addLayer(PoolingMaker::instance()->poolingSize(args.poolingSize));
        }
    }
    net->addLayer(FullyConnectedMaker::instance()->numPlanes(args.numCats)->imageSize(1)->biased());
    net->addLayer(SoftMaxMaker::instance());
    net->print();
    net->setBatchSize(args.batchSize);

    mt19937 random;
    random.seed(1); // so always gives same output
    const int inputsSize = net->getInputCubeSize() * args.batchSize;
    float *inputData = new float[ inputsSize ];
    for(int i = 0; i < inputsSize; i++) {
        inputData[i] = random() / (float)random.max() * 0.2f - 0.1f;
    }
//    const int outputNumElements = net->getLastLayer()->getOutputNumElements();
    int *labels = new int[args.batchSize];
    for(int i = 0; i < args.batchSize; i++) {
        labels[i] = random() % args.numCats;
    }

    for (int layerIndex = 1; layerIndex <= args.numLayers; layerIndex++) {
        ConvolutionalLayer *layer = dynamic_cast<ConvolutionalLayer*>(net->getLayer(layerIndex));
        if(layer != 0) {
            int weightsSize = layer->getWeightsSize();
    //        cout << "weightsSize, layer " << 1 << "= " << weightsSize << endl;
            for(int i = 0; i < weightsSize; i++) {
                layer->weights[i] = random() / (float)random.max() * 0.2f - 0.1f;
            }
            layer->weightsWrapper->copyToDevice();
            int biasSize = layer->getBiasSize();
            for(int i = 0; i < biasSize; i++) {
                layer->bias[i] = random() / (float)random.max() * 0.2f - 0.1f;
            }
        }
    }

    Timer timer;
    int weightsTotalSize = WeightsPersister::getTotalNumWeights(net);
    float *lastWeights = new float[weightsTotalSize];
    float *currentWeights = new float[weightsTotalSize];
    WeightsPersister::copyNetWeightsToArray(net, lastWeights);
//    cout << "learningRate: " << args.learningRate << endl;
    net->forward(inputData);
    float lastloss = 0;
    for(int i = 0; i < args.numEpochs; i++) {
//        net->learnBatch(args.learningRate, inputData, expectedOutput);
        net->forward(inputData);
        net->backwardFromLabels(labels);
        WeightsPersister::copyNetWeightsToArray(net, currentWeights);
        for(int layer = 1; layer < (int)net->getNumLayers(); layer++) {
            checkErrorsForLayer(layer, lastloss, net, lastWeights, currentWeights, args.learningRate, inputData, labels);
        }
        WeightsPersister::copyArrayToNetWeights(currentWeights, net);
        net->forward(inputData);
        float thisloss = net->calcLossFromLabels(labels);
        cout << "full thisloss: " << thisloss << endl;

//        float sumsquaredweightsdiff = sumWeightChangesSquared(lastWeights, currentWeights, net);
//        float lossChangeFromW = (sumsquaredweightsdiff/args.learningRate);
//        float thisloss = net->calcLossFromLabels(labels) ;
//        float lossChange = (lastloss - thisloss);
////        cout << "loss " << thisloss << " loss diff " << lossChange << endl;
//        if(i > 0) {
//            cout << "compare:" << endl;
//            cout << "    losschangefromw " << lossChangeFromW << endl;
//            cout << "    actual loss change " << lossChange << endl;
//            if(isnan(lossChange)) {
//                cout << "epoch " << i << endl;
//                cout << "compare:" << endl;
//                cout << "    losschangefromw " << lossChangeFromW << endl;
//                cout << "    actual loss change " << lossChange << endl;
//                EXPECT_TRUE(!isnan(lossChange));
//            }
//            if(lossChange / lossChangeFromW > 1.3f) {
//                cout << "epoch " << i << endl;
//                cout << "compare:" << endl;
//                cout << "    losschangefromw " << lossChangeFromW << endl;
//                cout << "    actual loss change " << lossChange << endl;
//                cout << "loss: " << lastloss << " -> " << thisloss << endl;
//                EXPECT_EQ(lossChange, lossChangeFromW);
//            } else if(lossChangeFromW / lossChange > 1.3f) {
//                cout << "epoch " << i << endl;
//                cout << "compare:" << endl;
//                cout << "    losschangefromw " << lossChangeFromW << endl;
//                cout << "    actual loss change " << lossChange << endl;
//                cout << "loss: " << lastloss << " -> " << thisloss << endl;
//                EXPECT_EQ(lossChange, lossChangeFromW);
//            }
//        }
        lastloss = thisloss;
//        memcpy(lastWeights1, currentWeights1, sizeof(float) * weights1Size);
        WeightsPersister::copyNetWeightsToArray(net, lastWeights);
    }
    timer.timeCheck("batch time");
    StatefulTimer::dump(true);

//    float *output = (float*)(net->getOutput());
//    Sampler::printSamples("net->getOutput()", outputNumElements, (float*)output);

    delete[]currentWeights;
    delete[]lastWeights;
    delete[] labels;
    delete[] inputData;
    delete net;
    delete cl;
}

TEST(testsinglebatch, imagesize5_filtersize3_batchsize2_softmax) {
    testLabelled(TestArgs::instance().BatchSize(2).LearningRate(0.001f).ImageSize(5).NumLayers(2)
            .FilterSize(3).NumFilters(5).NumEpochs(20).NumCats(5));
}

TEST(testsinglebatch, imagesize4_filtersize3_batchsize2_pooling) {
    testLabelled(TestArgs::instance().BatchSize(2).LearningRate(0.0001f).ImageSize(12).NumLayers(2)
            .NumFilters(5).FilterSize(3).NumEpochs(20).NumCats(5)
            .PoolingSize(2));
}

TEST(SLOW_testsinglebatch, imagesize4_filtersize3_batchsize2_pooling_args) {
    int numEpochs = 20;
    int poolingSize = 2;
    float learningRate = 0.01f;
    TestArgsParser::arg("numepochs", &numEpochs);
    TestArgsParser::arg("poolingsize", &poolingSize);
    TestArgsParser::arg("learningrate", &learningRate);
    TestArgsParser::go();
    testLabelled(TestArgs::instance().BatchSize(2).LearningRate(learningRate).ImageSize(12).NumLayers(2)
            .NumFilters(5).FilterSize(3).NumEpochs(numEpochs).NumCats(5)
            .PoolingSize(poolingSize));
}

/*
TEST(testsinglebatch, detailedregression) {
    const int batchSize = 128;
    const float learningRate = 0.1f;

    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(28)->instance();
    net->addLayer(ConvolutionalMaker::instance()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->addLayer(ConvolutionalMaker::instance()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->addLayer(ConvolutionalMaker::instance()->numFilters(10)->filterSize(20)->tanh()->biased()->insert();
    net->setBatchSize(batchSize);

    mt19937 random;
    random.seed(1); // so always gives same output
    const int inputsSize = net->getInputNumElementsPerExample() * batchSize;
    float *inputData = new float[ inputsSize ];
    for(int i = 0; i < inputsSize; i++) {
        inputData[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    const int outputNumElements = net->getLastLayer()->getOutputNumElements();
    float *expectedOutput = new float[outputNumElements];
    for(int i = 0; i < outputNumElements; i++) {
        expectedOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

    for (int layerIndex = 1; layerIndex <= 3; layerIndex++) {
        ConvolutionalLayer *layer = dynamic_cast<ConvolutionalLayer*>(net->layers[layerIndex]);
        int weightsSize = layer->getWeightsSize();
        for(int i = 0; i < weightsSize; i++) {
            layer->weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        }
        layer->weightsWrapper->copyToDevice();
        int biasSize = layer->getBiasSize();
        for(int i = 0; i < biasSize; i++) {
            layer->bias[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        }
//        layer->biasWrapper->copyToDevice();
    }

    net->forward(inputData);
    for (int layerIndex = 1; layerIndex <= 3; layerIndex++) {
        ConvolutionalLayer *layer = dynamic_cast<ConvolutionalLayer*>(net->layers[layerIndex]);
        float const*output = layer->getOutput();
        cout << "layer " << layerIndex << endl;
        Sampler::printSamples("output", outputNumElements, (float*)output, 3);        
    }
float *output = (float*)(net->layers[1]->getOutput());
EXPECT_FLOAT_NEAR(0.0767364, output[684]);
EXPECT_FLOAT_NEAR(0.0294366, output[559]);
EXPECT_FLOAT_NEAR(0.0244578, output[373]);
output = (float*)(net->layers[2]->getOutput());
EXPECT_FLOAT_NEAR(0.0867873, output[684]);
EXPECT_FLOAT_NEAR(0.0496127, output[559]);
EXPECT_FLOAT_NEAR(0, output[373]);
output = (float*)(net->layers[3]->getOutput());
EXPECT_FLOAT_NEAR(-0.232493, output[684]);
EXPECT_FLOAT_NEAR(0.179215, output[559]);
EXPECT_FLOAT_NEAR(0.14498, output[373]);

StatefulTimer::dump(true);

float *weights = net->layers[3]->weights;
int weightsSize = net->layers[3]->getWeightsSize();
Sampler::printSamples("weights", weightsSize, (float*)weights, 3);        

EXPECT_FLOAT_NEAR(-0.0688307, weights[16044]);
EXPECT_FLOAT_NEAR(0.0310387, weights[72239]);
EXPECT_FLOAT_NEAR(-0.0746839, weights[98933]);

cout << "=============== backprop ..." << endl;
//float *errors = new float[100000];
ConvolutionalLayer *layer3 = dynamic_cast<ConvolutionalLayer*>(net->layers[3]);
ExpectedValuesLayer *expectedValuesLayer = dynamic_cast<ExpectedValuesLayer*>(net->getLastLayer());
//ExpectedValuesLayer *expectedValuesLayer = (new ExpectedValuesLayerMaker(net, net->layers[3]))->instance();
expectedValuesLayer->setBatchSize(batchSize);
expectedValuesLayer->calcErrors(expectedOutput);
int layer3OutputNumElements = layer3->getOutputNumElements();
float *layer3errors = expectedValuesLayer->errors;
Sampler::printSamples("layer3errors", layer3OutputNumElements, layer3errors, 3);        

EXPECT_FLOAT_NEAR(-0.296495, layer3errors[684]);
EXPECT_FLOAT_NEAR(0.214934, layer3errors[559]);
EXPECT_FLOAT_NEAR(0.1246, layer3errors[373]);

cout << endl;

ConvolutionalLayer *layer2 = dynamic_cast<ConvolutionalLayer*>(net->layers[2]);
int layer2OutputNumElements = layer2->getOutputNumElements();
layer3->nextLayer = expectedValuesLayer;
layer3->backward(learningRate);
float *layer2errors = layer3->getDerivLossBySumForUpstream();
Sampler::printSamples("layer2errors", layer2OutputNumElements, layer2errors);

//EXPECT_FLOAT_NEAR(-0.296495, layer2errors[684]);
//EXPECT_FLOAT_NEAR(0.214934, layer2errors[559]);
//EXPECT_FLOAT_NEAR(0.1246, layer2errors[373]);

EXPECT_FLOAT_NEAR(0.00070503, layer2errors[1116844]);
EXPECT_FLOAT_NEAR(0.0220331, layer2errors[174639]);
EXPECT_FLOAT_NEAR(-0.00368077, layer2errors[1353333]);
EXPECT_FLOAT_NEAR(-0.0292891, layer2errors[314560]);
EXPECT_FLOAT_NEAR(0.0361823, layer2errors[176963]);

cout << endl;

ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer*>(net->layers[1]);
int layer1OutputNumElements = layer1->getOutputNumElements();
layer2->nextLayer = layer3;
layer2->backward(learningRate);
float *layer1errors = layer2->getDerivLossBySumForUpstream();
Sampler::printSamples("layer1errors", layer1OutputNumElements, layer1errors);

EXPECT_FLOAT_NEAR(-0.0137842, layer1errors[199340]);
EXPECT_FLOAT_NEAR(-0.015897, layer1errors[567855]);
EXPECT_FLOAT_NEAR(0.0170709, layer1errors[2270837]);

cout << endl;

layer1->nextLayer = layer2;
layer1->backward(learningRate);

//net->layers.push_back(expectedValuesLayer);
    net->backward(learningRate, expectedOutput);
    for (int layerIndex = 3; layerIndex >= 1; layerIndex--) {
        ConvolutionalLayer *layer = dynamic_cast<ConvolutionalLayer*>(net->layers[layerIndex]);
        weights = layer->weights;
        weightsSize = layer->getWeightsSize();
        cout << "layer " << layerIndex << endl;
        Sampler::printSamples("weights", weightsSize, (float*)weights, 3);        
        float *bias = layer->bias;
        int biasSize = layer->getBiasSize();
        cout << "layer " << layerIndex << endl;
        Sampler::printSamples("bias", biasSize, (float*)bias, 3);        
    }

layer3->weightsWrapper->copyToHost();
weights = net->layers[3]->weights;
EXPECT_FLOAT_NEAR(-0.0679266, weights[16044]);
EXPECT_FLOAT_NEAR(0.0284175, weights[72239]);
EXPECT_FLOAT_NEAR(-0.0749269, weights[98933]);
float *bias = net->layers[3]->bias;
EXPECT_FLOAT_NEAR(-0.0605856, bias[4]);
EXPECT_FLOAT_NEAR(-0.0663593, bias[9]);
EXPECT_FLOAT_NEAR(-0.0573801, bias[3]);
layer2->weightsWrapper->copyToHost();
weights = net->layers[2]->weights;
EXPECT_FLOAT_NEAR(0.0507008, weights[16044]);
EXPECT_FLOAT_NEAR(0.0982873, weights[21039]);
EXPECT_FLOAT_NEAR(-0.094224, weights[22133]);
bias = net->layers[2]->bias;
EXPECT_FLOAT_NEAR(-0.0552651, bias[12]);
EXPECT_FLOAT_NEAR(-0.0571462, bias[15]);
EXPECT_FLOAT_NEAR(-0.0304532, bias[21]);
layer1->weightsWrapper->copyToHost();
weights = net->layers[1]->weights;
EXPECT_FLOAT_NEAR(-0.0223737, weights[44]);
EXPECT_FLOAT_NEAR(-0.0658144, weights[239]);
EXPECT_FLOAT_NEAR(-0.0419252, weights[533]);
bias = net->layers[1]->bias;
EXPECT_FLOAT_NEAR(-0.0563513, bias[12]);
EXPECT_FLOAT_NEAR(-0.0601025, bias[15]);
EXPECT_FLOAT_NEAR(0.000941529, bias[21]);

output = (float*)(net->getOutput());
Sampler::printSamples("net->getOutput()", outputNumElements, (float*)output, 3);
EXPECT_FLOAT_NEAR(-0.232493, net->getOutput()[684]);
EXPECT_FLOAT_NEAR(0.179215, net->getOutput()[559]);
EXPECT_FLOAT_NEAR(0.14498, net->getOutput()[373]);



net->forward(inputData);
output = (float*)(net->getOutput());
Sampler::printSamples("net->getOutput()", outputNumElements, (float*)output, 3);

EXPECT_FLOAT_NEAR(0.549084, net->getOutput()[684]);
EXPECT_FLOAT_NEAR(-0.00702396, net->getOutput()[559]);
EXPECT_FLOAT_NEAR(-0.775789, net->getOutput()[373]);

//net->layers[1]->getOutput();

net->backward(learningRate, expectedOutput);

    for (int layerIndex = 3; layerIndex >= 1; layerIndex--) {
        ConvolutionalLayer *layer = dynamic_cast<ConvolutionalLayer*>(net->layers[layerIndex]);
        weights = layer->weights;
        weightsSize = layer->getWeightsSize();
        cout << "weights = net->layers[" << layerIndex << "]->weights;" << endl;
        Sampler::printSamples("weights", weightsSize, (float*)weights, 3);        
        float *bias = layer->bias;
        int biasSize = layer->getBiasSize();
        cout << "bias = net->layers[" << layerIndex << "]->bias;" << endl;
        Sampler::printSamples("bias", biasSize, (float*)bias, 3);        
    }

layer3->weightsWrapper->copyToHost();
weights = net->layers[3]->weights;
EXPECT_FLOAT_NEAR(-0.0681024, weights[16044]);
EXPECT_FLOAT_NEAR(0.0316504, weights[72239]);
EXPECT_FLOAT_NEAR(-0.0741202, weights[98933]);
bias = net->layers[3]->bias;
EXPECT_FLOAT_NEAR(-0.0986968, bias[4]);
EXPECT_FLOAT_NEAR(-0.0531305, bias[9]);
EXPECT_FLOAT_NEAR(-0.0268224, bias[3]);
layer2->weightsWrapper->copyToHost();
weights = net->layers[2]->weights;
EXPECT_FLOAT_NEAR(0.046951, weights[16044]);
EXPECT_FLOAT_NEAR(0.098209, weights[21039]);
EXPECT_FLOAT_NEAR(-0.0942325, weights[22133]);
bias = net->layers[2]->bias;
EXPECT_FLOAT_NEAR(-0.0552651, bias[12]);
EXPECT_FLOAT_NEAR(-0.0575521, bias[15]);
EXPECT_FLOAT_NEAR(-0.0281381, bias[21]);
layer1->weightsWrapper->copyToHost();
weights = net->layers[1]->weights;
EXPECT_FLOAT_NEAR(-0.0222835, weights[44]);
EXPECT_FLOAT_NEAR(-0.0658485, weights[239]);
EXPECT_FLOAT_NEAR(-0.0419671, weights[533]);
bias = net->layers[1]->bias;
EXPECT_FLOAT_NEAR(-0.0563201, bias[12]);
EXPECT_FLOAT_NEAR(-0.0600976, bias[15]);
EXPECT_FLOAT_NEAR(0.0122473, bias[21]);


    Timer timer;
    for(int i = 0; i < 2; i++) {
        net->learnBatch(learningRate, inputData, expectedOutput);
    }
    timer.timeCheck("batch time");
    StatefulTimer::dump(true);

    output = (float*)(net->getOutput());
    Sampler::printSamples("net->getOutput()", outputNumElements, (float*)output);

EXPECT_FLOAT_NEAR(-0.15081, net->getOutput()[684]);
EXPECT_FLOAT_NEAR(-0.0236106, net->getOutput()[559]);
EXPECT_FLOAT_NEAR(-0.0585419, net->getOutput()[373]);
EXPECT_FLOAT_NEAR(0.168737, net->getOutput()[960]);
EXPECT_FLOAT_NEAR(-0.00845184, net->getOutput()[323]);

    delete[] expectedOutput;
    delete[] inputData;
    delete net;
}

TEST(SLOW_testsinglebatch, perf) {
    const int batchSize = 128;
    const float learningRate = 0.1f;

    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(28)->instance();
    net->addLayer(ConvolutionalMaker::instance()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->addLayer(ConvolutionalMaker::instance()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->addLayer(ConvolutionalMaker::instance()->numFilters(10)->filterSize(20)->tanh()->biased()->insert();
//    ExpectedValuesLayer *expectedValuesLayer = (new ExpectedValuesLayerMaker(net, net->getLastLayer()))->instance();
//    net->getLastLayer()->nextLayer = expectedValuesLayer;
//    net->layers.push_back(expectedValuesLayer);
    net->setBatchSize(batchSize);

    mt19937 random;
    random.seed(1); // so always gives same output
    const int inputsSize = net->getInputNumElementsPerExample() * batchSize;
    float *inputData = new float[ inputsSize ];
    for(int i = 0; i < inputsSize; i++) {
        inputData[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    const int outputNumElements = net->getLastLayer()->getOutputNumElements();
    float *expectedOutput = new float[outputNumElements];
    for(int i = 0; i < outputNumElements; i++) {
        expectedOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

    for (int layerIndex = 1; layerIndex <= 3; layerIndex++) {
        ConvolutionalLayer *layer = dynamic_cast<ConvolutionalLayer*>(net->layers[layerIndex]);
        int weightsSize = layer->getWeightsSize();
        for(int i = 0; i < weightsSize; i++) {
            layer->weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        }
        layer->weightsWrapper->copyToDevice();
        int biasSize = layer->getBiasSize();
        for(int i = 0; i < biasSize; i++) {
            layer->bias[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        }
    }

    Timer timer;
    for(int i = 0; i < 5; i++) {
        net->learnBatch(learningRate, inputData, expectedOutput);
    }
    timer.timeCheck("batch time");
    StatefulTimer::dump(true);

    float *output = (float*)(net->getOutput());
    Sampler::printSamples("net->getOutput()", outputNumElements, (float*)output);

EXPECT_FLOAT_NEAR(-0.121662, net->getOutput()[684]);
EXPECT_FLOAT_NEAR(0.0783329, net->getOutput()[559]);
EXPECT_FLOAT_NEAR(-0.0549671, net->getOutput()[373]);
EXPECT_FLOAT_NEAR(0.0715649, net->getOutput()[960]);
EXPECT_FLOAT_NEAR(-0.00818501, net->getOutput()[323]);

    delete[] expectedOutput;
    delete[] inputData;
    delete net;
}

TEST(testsinglebatch, perf19) {
    const int batchSize = 128;
    const float learningRate = 0.1f;

    const int imageSize = 19;
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(imageSize)->instance();
    net->addLayer(ConvolutionalMaker::instance()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->addLayer(ConvolutionalMaker::instance()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->addLayer(ConvolutionalMaker::instance()->numFilters(10)->filterSize(imageSize - 4 * 2)->tanh()->biased()->insert();
//    ExpectedValuesLayer *expectedValuesLayer = (new ExpectedValuesLayerMaker(net, net->getLastLayer()))->instance();
//    net->getLastLayer()->nextLayer = expectedValuesLayer;
//    net->layers.push_back(expectedValuesLayer);
    net->setBatchSize(batchSize);

    mt19937 random;
    random.seed(1); // so always gives same output
    const int inputsSize = net->getInputNumElementsPerExample() * batchSize;
    float *inputData = new float[ inputsSize ];
    for(int i = 0; i < inputsSize; i++) {
        inputData[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    const int outputNumElements = net->getLastLayer()->getOutputNumElements();
    float *expectedOutput = new float[outputNumElements];
    for(int i = 0; i < outputNumElements; i++) {
        expectedOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

    for (int layerIndex = 1; layerIndex <= 3; layerIndex++) {
        ConvolutionalLayer *layer = dynamic_cast<ConvolutionalLayer*>(net->layers[layerIndex]);
        int weightsSize = layer->getWeightsSize();
        for(int i = 0; i < weightsSize; i++) {
            layer->weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        }
        layer->weightsWrapper->copyToDevice();
        int biasSize = layer->getBiasSize();
        for(int i = 0; i < biasSize; i++) {
            layer->bias[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        }
    }

    Timer timer;
    for(int i = 0; i < 5; i++) {
        net->learnBatch(learningRate, inputData, expectedOutput);
    }
    timer.timeCheck("batch time");
    StatefulTimer::dump(true);

    float *output = (float*)(net->getOutput());
    Sampler::printSamples("net->getOutput()", outputNumElements, (float*)output);

//EXPECT_FLOAT_NEAR(-0.121662, net->getOutput()[684]);
//EXPECT_FLOAT_NEAR(0.0783329, net->getOutput()[559]);
//EXPECT_FLOAT_NEAR(-0.0549671, net->getOutput()[373]);
//EXPECT_FLOAT_NEAR(0.0715649, net->getOutput()[960]);
//EXPECT_FLOAT_NEAR(-0.00818501, net->getOutput()[323]);

    delete[] expectedOutput;
    delete[] inputData;
    delete net;
}

TEST(SLOW_testsinglebatch, perf19_depth12) {
    const int batchSize = 128;
    const float learningRate = 0.1f;

    const int imageSize = 19;
    NeuralNet *net = NeuralNet::maker()->planes(32)->imageSize(imageSize)->instance();
    for(int i = 0; i < 12; i++) {
        net->addLayer(ConvolutionalMaker::instance()->numFilters(128)->filterSize(3)->relu()->padZeros()->biased()->insert();
    }
    net->addLayer(ConvolutionalMaker::instance()->numFilters(19*19)->filterSize(19)->tanh()->biased()->insert();
//    ExpectedValuesLayer *expectedValuesLayer = (new ExpectedValuesLayerMaker(net, net->getLastLayer()))->instance();
//    net->getLastLayer()->nextLayer = expectedValuesLayer;
//    net->layers.push_back(expectedValuesLayer);
    net->setBatchSize(batchSize);

    mt19937 random;
    random.seed(1); // so always gives same output
    const int inputsSize = net->getInputNumElementsPerExample() * batchSize;
    float *inputData = new float[ inputsSize ];
    for(int i = 0; i < inputsSize; i++) {
        inputData[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    const int outputNumElements = net->getLastLayer()->getOutputNumElements();
    float *expectedOutput = new float[outputNumElements];
    for(int i = 0; i < outputNumElements; i++) {
        expectedOutput[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

    for (int layerIndex = 1; layerIndex <= 3; layerIndex++) {
        ConvolutionalLayer *layer = dynamic_cast<ConvolutionalLayer*>(net->layers[layerIndex]);
        int weightsSize = layer->getWeightsSize();
        for(int i = 0; i < weightsSize; i++) {
            layer->weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        }
        layer->weightsWrapper->copyToDevice();
        int biasSize = layer->getBiasSize();
        for(int i = 0; i < biasSize; i++) {
            layer->bias[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        }
    }

    Timer timer;
    for(int i = 0; i < 1; i++) {
        net->learnBatch(learningRate, inputData, expectedOutput);
    }
    timer.timeCheck("batch time");
    StatefulTimer::dump(true);

    float *output = (float*)(net->getOutput());
    Sampler::printSamples("net->getOutput()", outputNumElements, (float*)output);

//EXPECT_FLOAT_NEAR(-0.121662, net->getOutput()[684]);
//EXPECT_FLOAT_NEAR(0.0783329, net->getOutput()[559]);
//EXPECT_FLOAT_NEAR(-0.0549671, net->getOutput()[373]);
//EXPECT_FLOAT_NEAR(0.0715649, net->getOutput()[960]);
//EXPECT_FLOAT_NEAR(-0.00818501, net->getOutput()[323]);

    delete[] expectedOutput;
    delete[] inputData;
    delete net;
}
*/

