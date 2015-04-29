# Architecture

* [NeuralNet.h](../src/NeuralNet.h) is a container for layers. It contains three types of method:
  * methods that iterate over each layer, eg `forward`
  * methods that call a method on the first layer, eg `getInputCubeSize`
  * methods that call a method on the last layer, eg `getOutput()`
* Various net layers, eg [ConvolutionalLayer.cpp](../src/ConvolutionalLayer.cpp), [PoolingLayer.cpp](../src/PoolingLayer.cpp), etc
* Trying to debug/unit-test by training whole layers is challenging, so the layer implementations are factorized, over two levels.  The first level abstracts away propagation, backprop of errors, and backprop of weights:
  * [Forward.cpp](../src/Forward.cpp) handles forward propagation
  * [Backward.cpp](../src/Backward.cpp) handles backward propagation of errors (strictly speaking: of the partial derivative of the loss with respect to the pre-activation sums for the layer)
    * The results of this layer are passed back through the stack of layers
  * [BackpropWeights2.cpp](../src/BackpropWeights2.cpp) handles backward propagation of weights, from the results of the appropriate Backward layer
* Then, each of these classes calls into implementation classes, which are children of the same class, which provide various kernels and implementations.  Eg, for [Forward.h](src/Forward.h], we have:
  * [Forward1.cpp](../src/Forward1.cpp)
  * [Forward2.cpp](../src/Forward2.cpp)
  * [Forward3.cpp](../src/Forward3.cpp)
  * ...
* ... and similarly for [Backward](../src/Backward.cpp), and [BackpropWeights2.cpp](../src/BackpropWeights2.cpp): each has implementation classes
* Therefore:
  * Testing can target one single implementation, or target only propagate or backproperrors, or backpropweights, rather than needing to test an entire network
  * These lower level factorized implementations could also plausibly be an appropriate unit of re-use
* There are also "meta"-layers, ie:
  * [ForwardAuto.cpp](../src/ForwardAuto.cpp): automatically tries different propagate kernels at run-time, and chooses the fastest :-)


