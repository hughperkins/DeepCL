#include "FullyConnectedLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

FullyConnectedLayer::FullyConnectedLayer( Layer *previousLayer, FullyConnectedMaker const *maker ) :
        Layer( previousLayer, maker ),
        errorsForUpstream(0),
        allocatedSize(0),
        activationFunction( maker->getActivationFunction() ) {
    std::cout << "WARNING: FullyConnectedLayer runs on CPU, and is  slllooowwww :-)" << std::endl;
    std::cout << "You'd better use a ConvolutionalLayer, and just make the filter size the same as " << std::endl;
    std::cout << "the upstream board size.  Use 'numFilters' to specify the number of output planes." << std::endl;
    std::cout << "ConvolutionalLayer will run significantly faster, since it runs on the GPU :-)" << std::endl;
    int fanIn = ( upstreamNumPlanes * upstreamBoardSize * upstreamBoardSize );
    weights = new float[ getWeightsSize() ];
    biasWeights = new float[ getBiasWeightsSize() ];
    randomizeWeights( fanIn, weights, getWeightsSize() );
    randomizeWeights( fanIn, biasWeights, getBiasWeightsSize() );
}
VIRTUAL float *FullyConnectedLayer::getResults() {
    return results;
}
VIRTUAL bool FullyConnectedLayer::needErrorsBackprop() {
    return true;
}
VIRTUAL ActivationFunction const*FullyConnectedLayer::getActivationFunction() {
    return activationFunction;
}
VIRTUAL int FullyConnectedLayer::getWeightsSize() const {
    return upstreamNumPlanes * upstreamBoardSize * upstreamBoardSize * numPlanes * boardSize * boardSize;
}
VIRTUAL int FullyConnectedLayer::getBiasWeightsSize() const {
    return numPlanes * boardSize * boardSize;
}
VIRTUAL void FullyConnectedLayer::print() const {
    std::cout << "FullyConnectedLayer" << std::endl;
    std::cout << "  numoutputneurons " << numPlanes << std::endl;
    printWeights();
    printOutput();
}
VIRTUAL void FullyConnectedLayer::printWeights() const {
    std::cout << "  weights: " << std::endl;
    for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
        for( int outRow = 0; outRow < boardSize; outRow++ ) {
            for( int outCol = 0; outCol < boardSize; outCol++ ) {
                std::cout << "    outPlane " << outPlane << " outrow " << outRow << " outCol " << outCol << std::endl;
                if( biased ) std::cout << "    bias: " << getBiasWeight( outPlane, outRow, outCol ) << std::endl;
                for( int inPlane = 0; inPlane < previousLayer->getNumPlanes(); inPlane++ ) {
                    if( previousLayer->getNumPlanes() > 1 ) std::cout << "    inPlane " << inPlane << std::endl;
                    for( int i = 0; i < std::min( 5, previousLayer->getBoardSize() ); i++ ) {
                        std::cout << "      ";
                        for( int j = 0; j < std::min(5, previousLayer->getBoardSize() ); j++ ) {
                           std::cout << getWeight( inPlane, i, j, outPlane, outRow, outCol ) << " ";
                        }
                        if( previousLayer->getBoardSize() > 5 ) {
                           std::cout << " ... " << " ";
                        }
                        std::cout << std::endl;
                    }
                }
                if( previousLayer->getBoardSize() > 5 ) {
                   std::cout << "       ... " << std::endl;
                }
                std::cout << std::endl;
            }
        }
    }
}
VIRTUAL void FullyConnectedLayer::printOutput() const {
    if( results == 0 ) {
         std::cout << "no results yet" << std::endl;
    } else {
        std::cout << "  outputs: " << std::endl;
        for( int n = 0; n < batchSize; n++ ) {
            std::cout << "    n: " << n << std::endl;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                if( numPlanes > 1 ) std::cout << "      plane " << plane << std::endl;
                if( boardSize == 1 ) {
                     std::cout << "        " << getResult(n, plane, 0, 0 ) << std::endl;
                } else {
                    std::cout << "not implemented for this boardsize" << std::endl;
                }
            }
        }
    }
}
void FullyConnectedLayer::randomizeWeights(int fanIn, float *weights, int numWeights ) {
//        std::cout << "fullyconnectedlayer randomzing weights" << std::endl;
    for( int i = 0; i < numWeights; i++ ) {
        weights[i] = generateWeight( fanIn );
    }
//        print();
}
VIRTUAL FullyConnectedLayer::~FullyConnectedLayer() {
//        if( results != 0 ) {
//            delete[] results;
//        }
//        delete[] weights;
}
VIRTUAL float *FullyConnectedLayer::getErrorsForUpstream() {
    return errorsForUpstream;
}
VIRTUAL void FullyConnectedLayer::setBatchSize( int batchSize ) {
    if( batchSize <= allocatedSize ) {
        this->batchSize = batchSize;
        return;
    }
    if( results != 0 ) {
        delete[] results;
    }
    if( errorsForUpstream != 0 ) { 
        delete[] errorsForUpstream;
    }
    this->batchSize = batchSize;
    results = new float[numPlanes * batchSize];
    if( layerIndex > 1 ) {
        errorsForUpstream = new float[ previousLayer->getResultsSize() ];
    }
    weOwnResults = true;
    allocatedSize = batchSize;
}
VIRTUAL void FullyConnectedLayer::propagate() {
    StatefulTimer::timeCheck("start fullyconnected propagate");
    for( int imageId = 0; imageId < batchSize; imageId++ ) {
        for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
            for( int outRow = 0; outRow < boardSize; outRow++ ) {
                for( int outCol = 0; outCol < boardSize; outCol++ ) {
                    float sum = 0;
                    for( int inPlane = 0; inPlane < upstreamNumPlanes; inPlane++ ) {
                        for( int inrow = 0; inrow < upstreamBoardSize; inrow++ ) {
                            for( int incol = 0; incol < upstreamBoardSize; incol++ ) {
                                float thisWeight = getWeight( inPlane, inrow, incol, outPlane, outRow, outCol );
                                float thisPixel = previousLayer->getResult( imageId, inPlane, inrow, incol );
    //std::cout << "weight " << thisWeight << " pixel " << thisPixel << std::endl;
                                sum += thisWeight * thisPixel;
                            }
                        }
                    }
                    if( biased ) {
                        sum += 0.5 * getBiasWeight( outPlane, outRow, outCol );
                    }
                    int resultIndex = getResultIndex( imageId, outPlane, outRow, outCol );
    //                results[numPlanes * imageId + out] = activationFn( sum );
                    results[resultIndex] = activationFunction->calc( sum );
    //                std::cout << "n " << imageId << " out " << out << " sum " << sum << " after actfn " << results[resultIndex] << std::endl;
                 }
            }
        }
    }
    StatefulTimer::timeCheck("end fullyconnected propagate");
}
// results structured like [imageid][outputplane][outputrow][outputcol]
//VIRTUAL void FullyConnectedLayer::calcErrors( float const *expected, float *errors ) {
//  //  backPropOld( learningRate, expected );
////        float *errors = new float[ batchSize * numPlanes * boardSize * boardSize ];
//    // matrix per-element subtraction...
//    for( int n = 0; n < batchSize; n++ ) {
//        for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
//            for( int outRow = 0; outRow < boardSize; outRow++ ) {
//                for( int outCol = 0; outCol < boardSize; outCol++ ) {
//                    int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
//                    errors[ resultIndex ] = results[resultIndex] - expected[resultIndex];
//                }
//            } 
//        }
//    }
////        backPropErrors( learningRate, errors );
////        delete[] errors;
//}
VIRTUAL void FullyConnectedLayer::backProp( float learningRate ) {
    throw std::runtime_error( "not (re-)implemented yet :-)");
/* todo :-)
    float *errors = nextLayer->getErrorsForUpstream();
    StatefulTimer::timeCheck("start fullyconnected backproperrors");
    for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
        for( int outRow = 0; outRow < boardSize; outRow++ ) {
            for( int outCol = 0; outCol < boardSize; outCol++ ) {
                for( int upstreamPlane = 0; upstreamPlane < upstreamNumPlanes; upstreamPlane++ ) {
                    for( int upstreamRow = 0; upstreamRow < upstreamBoardSize; upstreamRow++ ) {
                        for( int upstreamCol = 0; upstreamCol < upstreamBoardSize; upstreamCol++ ) {
                            float thiswchange = 0;
                            for( int n = 0; n < batchSize; n++ ) {
                                float upstreamResult = previousLayer->getResult( n, upstreamPlane, upstreamRow, upstreamCol );
                                int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                                float actualOutput = results[resultIndex];
                                float activationDerivative = activationFunction->calcDerivative( actualOutput );
                                float error = errors[resultIndex];
                                float thisimagethiswchange = upstreamResult * activationDerivative * error;
                                thiswchange += thisimagethiswchange;

                            }
                            int weightIndex = getWeightIndex( upstreamPlane, upstreamRow, upstreamCol, outPlane, outRow, outCol );
                            weights[ weightIndex ] -= learningRate * thiswchange / batchSize;
                        }
                    }
                }
               // handle bias...
               if( biased ) {
                   float thiswchange = 0;
                   for( int n = 0; n < batchSize; n++ ) {
                        float upstreamResult = 0.5;
                        int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                        float actualOutput = results[resultIndex];
                        float activationDerivative = activationFunction->calcDerivative( actualOutput );
                        float thisimagethiswchange = upstreamResult * errors[resultIndex] * activationDerivative;
                        thiswchange += thisimagethiswchange;
                   }
                   int biasWeightIndex = getBiasWeightIndex(outPlane, outRow, outCol);
                   biasWeights[ biasWeightIndex ] -= learningRate * thiswchange / batchSize;
               }
            }
        }
    }
    StatefulTimer::timeCheck("end fullyconnected backprop errors, start errors for upstream");
    if( layerIndex > 1 ) {
    for( int n = 0; n < batchSize; n++ ) {
        for( int upstreamPlane = 0; upstreamPlane < upstreamNumPlanes; upstreamPlane++ ) {
            for( int upstreamRow = 0; upstreamRow < upstreamBoardSize; upstreamRow++ ) {
                for( int upstreamCol = 0; upstreamCol < upstreamBoardSize; upstreamCol++ ) {
                    float sumWeightTimesOutError = 0;
                    for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
                        for( int outRow = 0; outRow < boardSize; outRow++ ) {
                            for( int outCol = 0; outCol < boardSize; outCol++ ) {
                                int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                                float thisError = errors[resultIndex];
                                int thisWeightIndex = getWeightIndex( upstreamPlane, upstreamRow, upstreamCol,
                                   outPlane, outRow, outCol );
                                float thisWeight = weights[thisWeightIndex];
                                float thisWeightTimesError = thisWeight * thisError;
                                sumWeightTimesOutError += thisWeightTimesError;
                            }
                        }
                    }
                    int upstreamResultIndex = previousLayer->getResultIndex( n, upstreamPlane, upstreamRow, upstreamCol );
                    errorsForUpstream[upstreamResultIndex] = sumWeightTimesOutError;
                }
            }
        }
    }
    }
    StatefulTimer::timeCheck("fully connected calc errors for usptream done");
    */
}



