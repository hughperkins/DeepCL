// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// declaration file for swig
// we might be able to reuse this for python wrappers too
// directors in swig are *much* easier than in Cython
// well... there are no directors in Cython, have to code them by hand :-P

%module(directors="1") luaDeepCL

%include "typemaps.i"
%include "carrays.i"
%include "std_string.i"

%{
#include "GenericLoader.h" // start with this first, since, if no data, kind of 
                           // hard to test things...
#include "NeuralNet.h"
#include "NetdefToNet.h"
#include "NetLearner.h"
#include "NormalizationLayerMaker.h"
#include "LayerMaker.h"
//#include "LuaWrappers.h"
%}

/*%typemap*/

/*// deepcl.GenericLoader_getDimensions*/
/*%rename(GenericLoader_getDimensions) mywrap_GenericLoader_getDimensions;*/
/*%inline %{*/
/*void mywrap_GenericLoader_getDimensions( std::string filepath, int *N_out, int *planes_out, int *size_out ) {*/
/*    std::cout << "wrap_GenericLoader_getDimensions" << std::endl;*/
/*/*    int N, planes, size;*/
/*    GenericLoader::getDimensions( filepath, N_out, planes_out, size_out );*/
/*    */
/*}*/
/*%}*/
/*//%rename(_GenericLoader_getDimensions) GenericLoader_getDimensions;*/
/*%ignore GenericLoader::getDimensions;*/
/*#define DeepCL_EXPORT*/
/*%include "GenericLoader.h"*/

/*%typemap*/

class GenericLoader {
public:
    static void getDimensions( std::string filepath, int *OUTPUT, int *OUTPUT, int *OUTPUT );
/*    static void load( std::string trainFilepath, unsigned char *INOUT, int *INOUT, int startN, int numExamples );*/
};

%inline %{
void GenericLoader_load( std::string trainFilepath, float *images, int *labels, int startN, int numExamples ) {
    int N, planes, size;
    GenericLoader::getDimensions( trainFilepath, &N, &planes, &size );
    int linearSize = numExamples * planes * size * size;
    // let's just convert to floats for now... since swig apparently makes floats easy, and
    // I'd need a bit of effort to persuade it to accept unsigned char * arrays plausibly
    unsigned char *ucarray = new unsigned char[linearSize];
    GenericLoader::load( trainFilepath, ucarray, labels, startN, numExamples );
    for( int i = 0; i < linearSize; i++ ) {
        images[i] = ucarray[i];
    }
    delete[] ucarray;
}
%}
void GenericLoader_load( std::string trainFilepath, float *INOUT, int *INOUT, int startN, int numExamples );

//class LayerMaker2 {
//public:
//};

class NeuralNet {
public:
    NeuralNet( int numPlanes, int imageSize );
    void addLayer( LayerMaker2 *maker );
    std::string asString();
};

class NetdefToNet {
public:
    static bool createNetFromNetdef( NeuralNet *net, std::string netdef );
};

%rename (NetLearnerBase) NetLearner;
template<typename T>
class NetLearner {
public:
    NetLearner( NeuralNet *net );
    void setTrainingData( int Ntrain, float *images, int *labels );
    void setTestingData( int Ntest, float *images, int *labels );
    void setSchedule( int numEpochs );
    void setBatchSize( int batchSize );
    void learn( float learningRate );
};

%template(NetLearnerFloats) NetLearner<float>;

// we can probalby just %include these actually, but writing 
// explicitly means we can pick and choose which methods we want
// and also tweak things easily
class NormalizationLayerMaker : public LayerMaker2 {
public:
    NormalizationLayerMaker();
    NormalizationLayerMaker *translate( float _translate );
    NormalizationLayerMaker *scale( float _scale );
};

// %apply float *IN { float *values };
//%apply int *OUT { p_numExamples };
//int *p_numExamples, int *p_numPlanes, int *p_imageSize
//%typemap 

%array_class(float, floatArray);  // this creates a brand-new lua type called 
                                  // `floatArray`, which represents a `float *`
                                  // in c++
                                  // we can construct it, and set values in it,
                                  // from lua, and pass it to c++, via swig 
                                  // wrappers
%array_class(unsigned char, unsignedCharArray);
%array_class(int, intArray);

