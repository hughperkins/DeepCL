// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// declaration file for swig
// we might be able to reuse this for python wrappers too
// directors in swig are *much* easier than in Cython
// well... there are no directors in Cython, have to code them by hand :-P
// hmmm, correction: directors arent supported in swig for lua :-P

%module(directors="1") LuaDeepCL

%include "typemaps.i"
%include "carrays.i"
%include "std_string.i"

%{
#include "DeepCL.h"
#include "EasyCL.h"
#include "qlearning/QLearner2.h"
%}

//%include "LuaWrappers.h"

class GenericLoader {
public:
    static void getDimensions( std::string filepath, int *OUTPUT, int *OUTPUT, int *OUTPUT );
};

%inline %{
void GenericLoader_load( std::string trainFilepath, float *images, int *labels, int startN, int numExamples ) {
    //int N, planes, size;
    //GenericLoader::getDimensions( trainFilepath, &N, &planes, &size );
    //int linearSize = numExamples * planes * size * size;
    // let's just convert to floats for now... since swig apparently makes floats easy, and
    // I'd need a bit of effort to persuade it to accept unsigned char * arrays plausibly
/*    unsigned char *ucarray = new unsigned char[linearSize];*/
    GenericLoader::load( trainFilepath, images, labels, startN, numExamples );
//    for( int i = 0; i < linearSize; i++ ) {
//        images[i] = ucarray[i];
 //   }
  //  delete[] ucarray;
}
%}
void GenericLoader_load( std::string trainFilepath, float *INOUT, int *INOUT, int startN, int numExamples );

//%ignore LayerMaker2;
//class LayerMaker2 {
//public:
//};

// class LayerMaker2;

class EasyCL {
public:
    static EasyCL *createForFirstGpu();
    static EasyCL *createForFirstGpuOtherwiseCpu();
    static EasyCL *createForIndexedGpu( int gpu );
    static EasyCL *createForPlatformDeviceIndexes(int platformIndex, int deviceIndex);
    static EasyCL *createForPlatformDeviceIds(cl_platform_id platformId, cl_device_id deviceId);
};

class TrainingContext {
public:
    TrainingContext( int epoch, int batch );
};

class Trainer {
public:
    virtual void train( NeuralNet *net, TrainingContext *context, float const*input, float const*expectedOutput ) = 0;
    virtual void trainFromLabels( NeuralNet *net, TrainingContext *context, float const*input, int const*labels ) = 0;
};

class BatchResult {
public:
    BatchResult( float loss, int numRight );
    float getLoss();
    float getNumRight();
};

class SGD : public Trainer {
public:
    static SGD *instance( EasyCL *cl, float learningRate );
    static SGD *instance( EasyCL *cl, float learningRate, float momentum );
    virtual void setMomentum( float momentum );
    virtual void setWeightDecay( float weightDecay );
    virtual std::string asString();
    virtual BatchResult train( NeuralNet *net, TrainingContext *context, float const*input, float const*expectedOutput );
    virtual BatchResult trainFromLabels( NeuralNet *net, TrainingContext *context, float const*input, int const*labels );
    SGD( EasyCL *cl );
};

class Annealer : public Trainer {
public:
    static Annealer *instance( EasyCL *cl, float learningRate, float anneal );
    Annealer( EasyCL *cl );
    virtual std::string asString();
    virtual void setAnneal( float anneal );
    virtual BatchResult train( NeuralNet *net, TrainingContext *context,
        float const*input, float const*expectedOutput );
    virtual BatchResult trainFromLabels( NeuralNet *net, TrainingContext *context,
        float const*input, int const*labels );
};

class Nesterov : public Trainer {
public:
    static Nesterov *instance( EasyCL *cl, float learningRate, float momentum );
    Nesterov( EasyCL *cl );
    virtual std::string asString();
    virtual void setMomentum( float momentum );
    virtual BatchResult train( NeuralNet *net, TrainingContext *context,
        float const*input, float const*expectedOutput );
    virtual BatchResult trainFromLabels( NeuralNet *net, TrainingContext *context,
        float const*input, int const*labels );
};

class NeuralNet {
public:
    NeuralNet( EasyCL *cl );
    NeuralNet( EasyCL *cl, int numPlanes, int imageSize );
    void addLayer( LayerMaker2 *maker );
    void setBatchSize( int batchSize );
    void forward( float const*images);
    void backwardFromLabels( int const *labels);
    void backward( float const *expectedOutput);
    int calcNumRight( int const *labels );
/*    int getOutputSize();*/
/*    float *getOutput();*/
    float const *getOutput() const;
    virtual int getOutputSize() const;
    //floatSlice getOutput();
    std::string asString();
    %extend {
        void getOutput( float *outputParam ) {
            int outputSize = $self->getOutputSize();
            float const*output = $self->getOutput();
            for( int i = 0; i < outputSize; i++ ) {
                outputParam[i] = output[i];
            }
        }
    }
};

class NetdefToNet {
public:
    static bool createNetFromNetdef( NeuralNet *net, std::string netdef );
};

/*%rename (NetLearnerBase) NetLearner;*/
class NetLearner {
public:
    NetLearner( Trainer *trainer, NeuralNet *net,
        int Ntrain, float *images, int *labels,
        int Ntest, float *images, int *labels,
        int batchSize );
    virtual void setSchedule( int numEpochs );
    virtual void setDumpTimings( bool dumpTimings );
    virtual void setSchedule( int numEpochs, int nextEpoch );
    virtual void reset();
    virtual bool tickBatch();  // just tick one learn batch, once all done, then run testing etc
    virtual bool getEpochDone();
    virtual int getNextEpoch();
    virtual int getNextBatch();
    virtual int getNTrain();
    virtual int getBatchNumRight();
    virtual float getBatchLoss();
    virtual void setBatchState( int nextBatch, int numRight, float loss );
    virtual bool tickEpoch();
    virtual void run();
    virtual bool isLearningDone();
};
/*%rename(NetLearner) NetLearnerFloats;*/
/*%template(NetLearner) NetLearner<float>;*/

// we can probalby just %include these actually, but writing 
// explicitly means we can pick and choose which methods we want
// and also tweak things easily
class NormalizationLayerMaker : public LayerMaker2 {
public:
    NormalizationLayerMaker();
    NormalizationLayerMaker *translate( float _translate );
    NormalizationLayerMaker *scale( float _scale );
};

/*%rename (InputLayerMakerBase) InputLayerMaker;*/

class InputLayerMaker : public LayerMaker2 {
public:
    InputLayerMaker();
    InputLayerMaker *numPlanes( int _numPlanes );
    InputLayerMaker *imageSize( int _imageSize );
};

class ActivationMaker : public LayerMaker2 {
public:
    ActivationMaker();
    ActivationMaker *relu();
    ActivationMaker *linear(); // nop basically
    ActivationMaker *tanh();
    ActivationMaker *sigmoid();
};

class DropoutMaker : public LayerMaker2 {
public:
    DropoutMaker();
    DropoutMaker *dropRatio( float _dropRatio );
};

class ConvolutionalMaker : public LayerMaker2 {
public:
    ConvolutionalMaker();
    ConvolutionalMaker *numFilters(int numFilters);
    ConvolutionalMaker *filterSize(int filterSize);
    ConvolutionalMaker *padZeros();
    ConvolutionalMaker *padZeros( bool value );
    ConvolutionalMaker *biased();
    ConvolutionalMaker *biased(int _biased);
};

class FullyConnectedMaker : public LayerMaker2 {
public:
    FullyConnectedMaker();
    FullyConnectedMaker *numPlanes(int numPlanes);
    FullyConnectedMaker *imageSize(int imageSize);
    FullyConnectedMaker *biased();
    FullyConnectedMaker *biased(int _biased);
};

class PoolingMaker : public LayerMaker2 {
public:
    PoolingMaker();
    PoolingMaker *poolingSize( int _poolingSize );
};

class SoftMaxMaker : public LayerMaker2 {
public:
    SoftMaxMaker();
    SoftMaxMaker *perColumn();
    SoftMaxMaker *perPlane();
};

class SquareLossMaker : public LayerMaker2 {
public:
    SquareLossMaker();
};

class CrossEntropyLossMaker : public LayerMaker2 {
public:
    CrossEntropyLossMaker();
};

%array_class(float, floatArray);  // this creates a brand-new lua type called 
                                  // `floatArray`, which represents a `float *`
                                  // in c++
                                  // we can construct it, and set values in it,
                                  // from lua, and pass it to c++, via swig 
                                  // wrappers
/*%array_class(unsigned char, unsignedCharArray);*/
%array_class(int, intArray);

class QLearner2 {
public:
    QLearner2( Trainer *trainer, NeuralNet *net, int numActions, int planes, int size );
    //QLearner2 *setPlanes( int planes );
    //QLearner2 *setSize( int size );
    //QLearner2 *setNumActions( int numActions );
    int step(double lastReward, bool wasReset, float *perception);
    void setLambda( float lambda );
    void setMaxSamples( int maxSamples );
    void setEpsilon( float epsilon );
    // void setLearningRate( float learningRate );
};

/*%inline %{
    void sliceArray( float *array, float *slice, int offset ) {
        slice = array + offset;
    }
%}

void sliceArray( float *array, float *OUTPUT, int offset );
*/

%define %array_slice(TYPE,NAME) // copy-pasted-hacked from carrays.i
                  // this will wrap a pointer, but not own it, ie shouldnt delete it
                  // we can create it from a carray, and an offset
         // for now, no accessors, just black box
%{
typedef TYPE NAME;
%}
typedef struct {
  /* Put language specific enhancements here */
} NAME;

%extend NAME {

NAME(TYPE *base, int offset) {
  return base + offset;
}
NAME(TYPE *base) { // will try using this for `float *getOutput()`
  return base;
}
~NAME() {
}

TYPE * cast() {
  return self;
}

};

%types(NAME = TYPE);

%enddef

%array_slice( float, floatSlice );
%array_slice( int, intSlice );

