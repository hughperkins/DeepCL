#include "InputLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

InputLayer::InputLayer( Layer *previousLayer, InputLayerMaker const*maker ) :
       Layer( previousLayer, maker ) {
}
VIRTUAL float *InputLayer::getResults() {
    return results;
}
VIRTUAL void InputLayer::printOutput() const {
    if( results == 0 ) {
         return;
    }
    for( int n = 0; n < std::min(5,batchSize); n++ ) {
        std::cout << "InputLayer n " << n << ":" << std::endl;
        for( int plane = 0; plane < std::min( 5, numPlanes); plane++ ) {
            if( numPlanes > 1 ) std::cout << "    plane " << plane << ":" << std::endl;
            for( int i = 0; i < std::min(5,boardSize); i++ ) {
                std::cout << "      ";
                for( int j = 0; j < std::min(5,boardSize); j++ ) {
                    std::cout << getResult( n, plane, i, j ) << " ";
//results[
//                            n * numPlanes * boardSize*boardSize +
//                            plane*boardSize*boardSize +
//                            i * boardSize +
//                            j ] << " ";
                }
                if( boardSize > 5 ) std::cout << " ... ";
                std::cout << std::endl;
            }
            if( boardSize > 5 ) std::cout << " ... " << std::endl;
        }
        if( numPlanes > 5 ) std::cout << "   ... other planes ... " << std::endl;
    }
    if( batchSize > 5 ) std::cout << "   ... other n ... " << std::endl;
}
VIRTUAL void InputLayer::print() const {
    printOutput();
}
void InputLayer::in( float const*images ) {
//        std::cout << "InputLayer::in()" << std::endl;
    this->results = (float*)images;
//        this->batchStart = batchStart;
//        this->batchEnd = batchEnd;
//        print();
}
VIRTUAL InputLayer::~InputLayer() {
}
VIRTUAL bool InputLayer::needErrorsBackprop() {
    return false;
}
VIRTUAL void InputLayer::setBatchSize( int batchSize ) {
//        std::cout << "inputlayer setting batchsize " << batchSize << std::endl;
    this->batchSize = batchSize;
}
VIRTUAL void InputLayer::propagate() {
}
VIRTUAL void InputLayer::backPropErrors( float learningRate, float const *errors ) {
}

