#pragma once

#include <iostream>
#include <string>

class NeuralNet;

#define VIRTUAL virtual
#define STATIC static

// whilst this class is portable, the weights files created totally are not (ie: endianness)
// but okish for now... (since it's not like weights files tend to be shared around much, and
// if they are, then the quickly-written file created by this could be converted by another
// utility into a portable datafile
// target usage for this class:
// - quickly snapshotting the weights after each epoch, therefore should be:
//    - fast, low IO :-)
class WeightsPersister {
public:
    
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    template< typename T > STATIC void copyArray( T *dst, T const*src, int length );  // this might already be in standard C++ library?
    STATIC int getTotalNumWeights( NeuralNet *net );
    STATIC void copyNetWeightsToArray( NeuralNet *net, float *target );
    STATIC void copyArrayToNetWeights( float const*source, NeuralNet *net );
    STATIC void persistWeights( std::string filepath, std::string trainingConfigString, NeuralNet *net, int epoch, int batch, float annealedLearningRate, int numRight, float loss );
    STATIC bool loadWeights( std::string filepath, std::string trainingConfigString, NeuralNet *net, int *p_epoch, int *p_batch, float *p_annealedLearningRate, int *p_numRight, float *p_loss );

    // [[[end]]]
};

