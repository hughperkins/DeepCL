<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Architecture](#architecture)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Architecture

* [NeuralNet.h](src/NeuralNet.h) is a container for layers. It contains three types of method:
  * methods that iterate over each layer, eg `propagate`
  * methods that call a method on the first layer, eg `getInputCubeSize`
  * methods that call a method on the last layer, eg `getResults()`
* Various net layers, eg [ConvolutionalLayer.cpp](src/ConvolutionalLayer.cpp), [PoolingLayer.cpp](src/PoolingLayer.cpp), etc
* Trying to debug/unit-test by training whole layers is challenging, so the layer implementations are factorized, over two levels.  The first level abstracts away propagation, backprop of errors, and backprop of weights:
  * [Propagate.cpp](src/Propagate.cpp) handles forward propagation
  * [BackpropErrorsv2.cpp](src/BackpropErrorsv2.cpp) handles backward propagation of errors (strictly speaking: of the partial derivative of the loss with respect to the pre-activation sums for the layer)
    * The results of this layer are passed back through the stack of layers
  * [BackpropWeights2.cpp](src/BackpropWeights2.cpp) handles backward propagation of weights, from the results of the appropriate BackpropErrorsv2 layer
* Then, each of these classes calls into implementation classes, which are children of the same class, which provide various kernels and implementations.  Eg, for [Propagate.h](src/Propagate.h], we have:
  * [Propagate1.cpp](src/Propagate1.cpp)
  * [Propagate2.cpp](src/Propagate2.cpp)
  * [Propagate3.cpp](src/Propagate3.cpp)
  * ...
* ... and similarly for [BackpropErrorsv2](src/BackpropErrorsv2.cpp), and [BackpropWeights2.cpp](src/BackpropWeights2.cpp): each has implementation classes
* Therefore:
  * Testing can target one single implementation, or target only propagate or backproperrors, or backpropweights, rather than needing to test an entire network
  * These lower level factorized implementations could also plausibly be an appropriate unit of re-use
* There are also "meta"-layers, ie:
  * [PropagateAuto.cpp](src/PropagateAuto.cpp): automatically tries different propagate kernels at run-time, and chooses the fastest :-)


