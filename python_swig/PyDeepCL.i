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

%module PyDeepCL

%include "typemaps.i"
%include "carrays.i"
%include "std_string.i"

%{
#define SWIG_FILE_WITH_INIT
%}
%include "thirdparty/numpy/numpy.i"
%init %{
import_array();
%}

%{
#include "GenericLoader.h" // start with this first, since, if no data, kind of 
                           // hard to test things...
#include "NeuralNet.h"
#include "NetdefToNet.h"
#include "NetLearner.h"
#include "NormalizationLayerMaker.h"
#include "LayerMaker.h"
#include "InputLayerMaker.h"
//#include "LuaWrappers.h"
#include "QLearner2.h"
%}

//%include "LuaWrappers.h"

class GenericLoader {
public:
    static void getDimensions( std::string filepath, int *OUTPUT, int *OUTPUT, int *OUTPUT );
    %extend {
        static void load2( std::string imagesFilePath, float *ARGOUT_ARRAY1, int DIM1, int *labels, int startN, int numExamples ) {
            GenericLoader::load( imagesFilePath, ARGOUT_ARRAY1, labels, startN, numExamples );
        }
    }
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
//%inline %{
class LayerMaker2 {
public:
    LayerMaker2();
    virtual ~LayerMaker2() {}
    virtual LayerMaker2 *clone() const = 0; // we have to mark this class pure abstract, otherwise
                              // things dont build.  shouldnt call this method though
//    virtual Layer *createLayer( Layer *previousLayer ) = 0;
};
//%}

// class LayerMaker2;

class NeuralNet {
public:
    NeuralNet();
    NeuralNet( int numPlanes, int imageSize );
    void addLayer( LayerMaker2 *maker );
    void setBatchSize( int batchSize );
    void propagate( float const*images);
    void backPropFromLabels( float learningRate, int const *labels);
    void backProp( float learningRate, float const *expectedResults);
    int calcNumRight( int const *labels );
/*    int getResultsSize();*/
/*    float *getResults();*/
    float const *getResults() const;
    virtual int getResultsSize() const;
    //floatSlice getResults();
    std::string asString();
    %extend {
        void getResults( float *resultsParam ) {
            int resultsSize = $self->getResultsSize();
            float const*results = $self->getResults();
            for( int i = 0; i < resultsSize; i++ ) {
                resultsParam[i] = results[i];
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
    NetLearner( NeuralNet *net,
        int Ntrain, float *images, int *labels,
        int Ntest, float *images, int *labels,
        int batchSize );
    void setSchedule( int numEpochs );
/*    void setBatchSize( int batchSize );*/
    void learn( float learningRate );
};

// not sure why we need these, but segfaults without these :-(
%typemap(ret) InputLayerMaker* %{ 
    result = result-> clone();
resultobj = SWIG_NewPointerObj(SWIG_as_voidptr(result), SWIGTYPE_p_InputLayerMaker, 0 |  0 );
%}
%typemap(ret) NormalizationLayerMaker* %{ 
    result = result-> clone();
resultobj = SWIG_NewPointerObj(SWIG_as_voidptr(result), SWIGTYPE_p_NormalizationLayerMaker, 0 |  0 );
%}
%typemap(ret) ConvolutionalMaker* %{ 
    result = result-> clone();
resultobj = SWIG_NewPointerObj(SWIG_as_voidptr(result), SWIGTYPE_p_ConvolutionalMaker, 0 |  0 );
%}
%typemap(ret) FullyConnectedMaker* %{ 
    result = result-> clone();
resultobj = SWIG_NewPointerObj(SWIG_as_voidptr(result), SWIGTYPE_p_FullyConnectedMaker, 0 |  0 );
%}
%typemap(ret) PoolingMaker* %{ 
    result = result-> clone();
resultobj = SWIG_NewPointerObj(SWIG_as_voidptr(result), SWIGTYPE_p_PoolingMaker, 0 |  0 );
%}
%typemap(ret) SoftMaxMaker* %{ 
    result = result-> clone();
resultobj = SWIG_NewPointerObj(SWIG_as_voidptr(result), SWIGTYPE_p_SoftMaxMaker, 0 |  0 );
%}
%typemap(ret) SquareLossMaker* %{ 
    result = result-> clone();
resultobj = SWIG_NewPointerObj(SWIG_as_voidptr(result), SWIGTYPE_p_SquareLossMaker, 0 |  0 );
%}
%typemap(ret) CrossEntropyLossMaker* %{ 
    result = result-> clone();
resultobj = SWIG_NewPointerObj(SWIG_as_voidptr(result), SWIGTYPE_p_CrossEntropyLossMaker, 0 |  0 );
%}


// we can probalby just %include these actually, but writing 
// explicitly means we can pick and choose which methods we want
// and also tweak things easily
class NormalizationLayerMaker : public LayerMaker2 {
public:
//    NormalizationLayerMaker( float _translate, float _scale );
    NormalizationLayerMaker();
    NormalizationLayerMaker *translate( float _translate );
    NormalizationLayerMaker *scale( float _scale );
    virtual NormalizationLayerMaker *clone() const;
//    virtual Layer *createLayer( Layer *previousLayer );
    %extend {
        float getTranslate(){ 
            return $self->_translate; 
        }
        float getScale(){ 
            return $self->_scale; 
        }
        void show(){ 
            std::cout << "normalizationlayermaker scale=" << $self->_scale << " translate=" << $self->_translate
                << std::endl; 
        }
    }
};


/*%rename (InputLayerMakerBase) InputLayerMaker;*/

class InputLayerMaker : public LayerMaker2 {
public:
    InputLayerMaker();
    InputLayerMaker *numPlanes( int _numPlanes );
    InputLayerMaker *imageSize( int _imageSize );
    virtual NormalizationLayerMaker *clone() const;
//    virtual Layer *createLayer( Layer *previousLayer );
    %extend {
        int getPlanes(){ 
            return $self->_numPlanes; 
        }
        int getSize(){ 
            return $self->_imageSize; 
        }
        void show(){ 
            std::cout << "InputLayerMaker numPlanes=" << $self->_numPlanes << " imageSize=" << $self->_imageSize
                << std::endl; 
        }
    }
};
/*%template(InputLayerMaker) InputLayerMaker<float>;*/

class ConvolutionalMaker : public LayerMaker2 {
public:
    ConvolutionalMaker();
    ConvolutionalMaker *numFilters(int numFilters);
    ConvolutionalMaker *filterSize(int filterSize);
    ConvolutionalMaker *padZeros();
    ConvolutionalMaker *padZeros( bool value );
    ConvolutionalMaker *biased();
    ConvolutionalMaker *biased(int _biased);
    ConvolutionalMaker *tanh();
    ConvolutionalMaker *relu();
    ConvolutionalMaker *sigmoid();
    ConvolutionalMaker *linear();
    virtual NormalizationLayerMaker *clone() const;
//    virtual Layer *createLayer( Layer *previousLayer );
};

class FullyConnectedMaker : public LayerMaker2 {
public:
    FullyConnectedMaker();
    FullyConnectedMaker *numPlanes(int numPlanes);
    FullyConnectedMaker *imageSize(int imageSize);
    FullyConnectedMaker *biased();
    FullyConnectedMaker *biased(int _biased);
    FullyConnectedMaker *linear();
    FullyConnectedMaker *tanh();
    FullyConnectedMaker *sigmoid();
    FullyConnectedMaker *relu();
    virtual NormalizationLayerMaker *clone() const;
//    virtual Layer *createLayer( Layer *previousLayer );
};

class PoolingMaker : public LayerMaker2 {
public:
    PoolingMaker();
    PoolingMaker *poolingSize( int _poolingSize );
    virtual NormalizationLayerMaker *clone() const;
//    virtual Layer *createLayer( Layer *previousLayer );
};

class SoftMaxMaker : public LayerMaker2 {
public:
    SoftMaxMaker();
    SoftMaxMaker *perColumn();
    SoftMaxMaker *perPlane();
    virtual NormalizationLayerMaker *clone() const;
//    virtual Layer *createLayer( Layer *previousLayer );
};

class SquareLossMaker : public LayerMaker2 {
public:
    SquareLossMaker();
    virtual NormalizationLayerMaker *clone() const;
//    virtual Layer *createLayer( Layer *previousLayer );
};

class CrossEntropyLossMaker : public LayerMaker2 {
public:
    CrossEntropyLossMaker();
    virtual NormalizationLayerMaker *clone() const;
//    virtual Layer *createLayer( Layer *previousLayer );
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
    QLearner2( NeuralNet *net, int numActions, int planes, int size );
    //QLearner2 *setPlanes( int planes );
    //QLearner2 *setSize( int size );
    //QLearner2 *setNumActions( int numActions );
    int step(double lastReward, bool wasReset, float *perception);
    void setLambda( float lambda );
    void setMaxSamples( int maxSamples );
    void setEpsilon( float epsilon );
    void setLearningRate( float learningRate );
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
NAME(TYPE *base) { // will try using this for `float *getResults()`
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

