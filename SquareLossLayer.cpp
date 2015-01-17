#include "SquareLossLayer.h"
#include "LossLayer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

SquareLossLayer::SquareLossLayer( Layer *previousLayer, SquareLossMaker const*maker ) :
        LossLayer( previousLayer, maker ),
        derivLossBySum( 0 ),
        allocatedSize( 0 ) {
}
VIRTUAL SquareLossLayer::~SquareLossLayer(){
    if( derivLossBySum != 0 ) {
        delete[] derivLossBySum;
    }
}
VIRTUAL float*SquareLossLayer::getDerivLossBySum() {
    return derivLossBySum;
}
//VIRTUAL float*SquareLossLayer::getDerivLossBySumForUpstream() {
//    return derivLossBySum;
//}
VIRTUAL float SquareLossLayer::calcLoss( float const *expected ) {
    float loss = 0;
    float *results = getResults();
    cout << "SquareLossLayer::calcLoss" << endl;
    // this is matrix subtraction, then element-wise square, then aggregation
    int numPlanes = previousLayer->getOutputPlanes();
    int boardSize = previousLayer->getOutputBoardSize();
    for( int imageId = 0; imageId < batchSize; imageId++ ) {
        for( int plane = 0; plane < numPlanes; plane++ ) {
            for( int outRow = 0; outRow < boardSize; outRow++ ) {
                for( int outCol = 0; outCol < boardSize; outCol++ ) {
                    int resultOffset = ( ( imageId
                         * numPlanes + plane )
                         * boardSize + outRow )
                         * boardSize + outCol;
 //                   int resultOffset = getResultIndex( imageId, plane, outRow, outCol ); //imageId * numPlanes + out;
                    float expectedOutput = expected[resultOffset];
                    float actualOutput = results[resultOffset];
                    float diff = actualOutput - expectedOutput;
                    float squarederror = diff * diff;
                    loss += squarederror;
                }
            }
        }            
    }
    loss *= 0.5f;
    cout << "loss " << loss << endl;
    return loss;
 }
VIRTUAL void SquareLossLayer::setBatchSize( int batchSize ) {
    if( batchSize <= allocatedSize ) {
        this->batchSize = batchSize;
        return;
    }
    if( derivLossBySum != 0 ) {
        delete[] derivLossBySum;
    }
    derivLossBySum = new float[ batchSize * previousLayer->getResultsSize() ];
    this->batchSize = batchSize;
    allocatedSize = batchSize;
}
VIRTUAL void SquareLossLayer::calcDerivLossBySum( float const*expectedResults ) {
    ActivationFunction const*fn = previousLayer->getActivationFunction();
    int resultsSize = previousLayer->getResultsSize();
    float *results = previousLayer->getResults();
    for( int i = 0; i < resultsSize; i++ ) {
        float result = results[i];
        float partialOutBySum = fn->calcDerivative( result );
        float partialLossByOut = result - expectedResults[i];
        derivLossBySum[i] = partialLossByOut * partialOutBySum;
    }
}

