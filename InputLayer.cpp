#include "InputLayer.h"

using namespace std;

InputLayer::InputLayer( Layer *previousLayer, InputLayerMaker const*maker ) :
       Layer( previousLayer, maker ) {
}
// [virtual]
float *InputLayer::getResults() {
    return results;
}
// [virtual]
void InputLayer::printOutput() const {
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
// [virtual]
void InputLayer::print() const {
    printOutput();
}
void InputLayer::in( float const*images ) {
//        std::cout << "InputLayer::in()" << std::endl;
    this->results = (float*)images;
//        this->batchStart = batchStart;
//        this->batchEnd = batchEnd;
//        print();
}
// [virtual]
InputLayer::~InputLayer() {
}
// [virtual]
bool InputLayer::needErrorsBackprop() {
    return false;
}
// [virtual]
void InputLayer::setBatchSize( int batchSize ) {
//        std::cout << "inputlayer setting batchsize " << batchSize << std::endl;
    this->batchSize = batchSize;
}
// [virtual]
void InputLayer::propagate() {
}
// [virtual]
void InputLayer::backPropErrors( float learningRate, float const *errors ) {
}

