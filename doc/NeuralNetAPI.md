<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Neural Net API](#neural-net-api)
  - [Create a net](#create-a-net)
  - [Add an input layer](#add-an-input-layer)
  - [Normalization layer](#normalization-layer)
  - [Dropout layer](#dropout-layer)
  - [Random patch layer](#random-patch-layer)
  - [Random translations layer](#random-translations-layer)
  - [Convolutional layers](#convolutional-layers)
  - [Activation layers](#activation-layers)
  - [Fully connected layers](#fully-connected-layers)
  - [Max-pooling layers](#max-pooling-layers)
  - [Loss layer](#loss-layer)
  - [Data format](#data-format)
  - [Create a Trainer](#create-a-trainer)
  - [Train](#train)
  - [Test](#test)
  - [Weight initialization](#weight-initialization)
  - [More details](#more-details)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Neural Net API

* You can create a network in C++ directly.  As an example, to create a `8C5-MP2-16C5-MP3-150N-10N` network, for MNIST, you could do:
```c++
EasyCL *cl = new EasyCL();
NeuralNet *net = new NeuralNet(cl);
net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(28) );
net->addLayer( NormalizationLayerMaker::instance()->translate( -mean )->scale( 1.0f / standardDeviation ) );
net->addLayer( ConvolutionalMaker::instance()->numFilters(8)->filterSize(5)->biased() );
net->addLayer( ActivationMaker::instance()->relu() );
net->addLayer( PoolingMaker::instance()->poolingSize(2) );
net->addLayer( ConvolutionalMaker::instance()->numFilters(16)->filterSize(5)->biased() );
net->addLayer( ActivationMaker::instance()->relu() );
net->addLayer( PoolingMaker::instance()->poolingSize(3) );
net->addLayer( FullyConnectedMaker::instance()->numPlanes(150)->imageSize(1)->biased() );
net->addLayer( ActivationMaker::instance()->relu() );
net->addLayer( FullyConnectedMaker::instance()->numPlanes(10)->imageSize(1)->biased() );
net->addLayer( ActivationMaker::instance()->linear() );
net->addLayer( SoftMaxMaker::instance() );
net->print();
```
* The following sections will detail the various layers available, and the options available for each layer type
* Data must be provided in contiguous, 1d format, see below

## Create a net

```c++
#include "DeepCL.h"
OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
NeuralNet *net = new NeuralNet( cl );
```

## Add an input layer

* You need exactly one input layer:
```c++
net->addLayer( InputMaker::instance()->numPlanes(10)->imageSize(19) );
```
* You need to set the number of input planes, and the image size.

## Normalization layer

* You can add a normalization layer, to translate and scale input data.  Put it just after the input layer, like this:
```c++
NeuralNet *net = new NeuralNet();
net->addLayer( InputMaker::instance()->numPlanes(10)->imageSize(19) );
net->addLayer( NormalizationMaker::instance()->translate( - mean )->scale( 1.0f / standardDeviation ) );
// other layers here...
```

## Dropout layer

To add a drop out layer:
```c++
net->addLayer( DropoutMaker::instance()->dropRatio(0.5f) );
```

This should probably go in between a fully-connected layer, and its associated activation layer, like:
```c++
net->addLayer( FullyConnectedMaker::instance()->numPlanes(10)->imageSize(1)->linear()->biased() );
net->addLayer( DropoutMaker::instance()->dropRatio(0.5f) );
net->addLayer( ActivationMaker::instance()->tanh() );
```

## Random patch layer

* You can add a random patch layer, to cut a patch from each image, in a random location, and train against that
* You need to specify the patch size, eg on minst, which is 28x28 images, you might use a patch size of 24
* During training the patch location is chosen randomly, per image, per epoch
* Size of output image from this layer is the size of the patch
* During testing, the patch is cut from the centre of the image
```c++
net->addLayer( RandomPatchMaker::instance()->patchSize(24) );
```

## Random translations layer

* You can add a random translations layer, to randomly translate each input image by a random amount, during training
* During testing, no translation is done
* If you put eg `translateSize(2)`, then the translation amount will be chosen uniformly from the set `{-2,-1,0,1,2}`, for each axis.
* Output image from this layer is same size as input image
```c++
net->addLayer( RandomTranslationsMaker::instance()->translateSize(2) );
```

## Convolutional layers

Eg:
```c++
net->addLayer( ConvolutionalMaker::instance()->numFilters(32)->filterSize(5)->relu()->biased() );
```

* You can change the number of filters, and their size.  If you want, you can use any of the following options:
  * `->padZeros()`: pad the input image with zeros, so the output image is same size as the input
  * `->biased()` turn on bias
  * `->biased(1)` same as `->biased()`
  * `->biased(0)` turn off bias (default)
* convolutional layers forward-prop and backward-prop both run on GPU, via OpenCL

## Activation layers

Eg:
```c++
net->addLayer( ActivationMaker::instance()->relu() );
```

* You can create one of the following activations to be applied on the previous layer.
  * `->linear()` choose linear activation
  * `->relu()` choose RELU activation
  * `->elu()` choose ELU activation
  * `->sigmoid()` choose sigmoid activation
  * `->tanh()` choose tanh activation (current default, but defaults can change...)
  * `->scaledtanh()` `1.7159 * tanh(0.66667 * x )`

## Fully connected layers

eg:
```c++
net->addLayer( FullyConnectedMaker::instance()->numPlanes(2)->imageSize(28) );
```

Available options:
  * `->biased()` turn on bias
  * `->biased(1)` same as `->biased()`
  * `->biased(0)` turn off bias (default)
  * `->linear()` choose linear activation
  * `->relu()` choose relu activation
  * `->sigmoid()` choose sigmoid activation
  * `->tanh()` choose tanh activation (current default, but defaults can change...)
  * `->scaledtanh()` `1.7159 * tanh(0.66667 * x )`

## Max-pooling layers

```c++
net->addLayer( PoolingMaker::instance()->poolingSize(2) );
```
* By default, if the input image size is not an exact multiple of the poolingsize, the extra margin will be ignored
* You can specify `padZeros` to include this margin:
```c++
net->addLayer( PoolingMaker::instance()->poolingSize(2)->padZeros() );
```

## Loss layer

You need to add exactly one loss layer, as the last layer of the net.  The following loss layers are available:
```c++
net->addLayer( SquareLossMaker::instance() );
net->addLayer( CrossEntropyMaker::instance() );
net->addLayer( SoftMaxLayer::instance() );
```
* if your outputs are categorial, 1-of-N, then softMaxLayer is probably what you want
* otherwise, you can choose square loss, or cross-entropy loss:
  * squared loss works well with a `tanh` last layer
  * cross entropy loss works well with a `sigmoid` last layer
  * if you're not sure, then `tanh` last layer, with squared loss, works well
* the softmax layer:
  * creates a probability distribution, ie a set of outputs, that sum to 1, and each lie in the range `0 <= x <= 1`
  * can create this probability distribution either across all output planes, with a imagesize of 1
    * this is the default
  * or else a per-plane probability distribution
    * add option `->perPlane()`

## Data format

Input data should be provided in a contiguous array, of `float`s.  "group by" order should be:

* training example id
* input plane
* image row
* image column

Providing labels, as an integer array, is the most efficient way of training, if you are training against categorical data.  The labels should be provided as one integer per example, zero-based.

* in this case, the last layer of the net should have the same number of nodes as categories, eg a `netdef` ending in `-5n`, if there are 5 categories
* if using the C++ API, you would probably want to use a `softmax` loss layer

For non-categorical data, you can provide expected output values as a contiguous array of floats. "group by" order for the floats should be:

* training example id
* output plane (eg, corresponds to filter id, for convolutional network)
* output row
* output column

## Create a Trainer

```c++
// create a Trainer object, currently SGD,
// passing in learning rate, and momentum:
Trainer *trainer = SGD::instance( cl, 0.02f, 0.0f );
```

Can set weightdecay, momentum, learningrate:

```c++
SGD *sgd = SGD::instance( cl );
sgd->setLearningRate( 0.002f );
sgd->setMomentum( 0.1f );
sgd->setWeightDecay( 0.001f );
```

Other trainers:
```c++
Adagrad *adagrad = new Adagrad( cl );
adagrad->setLearningRate( 0.002f );
Trainer *trainer = adagrad;

Rmsprop *rmsprop = new Rmsprop( cl );
rmsprop->setLearningRate( 0.002f );
Trainer *trainer = rmsprop;

Nesterov *nesterov = new Nesterov( cl );
nesterov->setLearningRate( 0.002f );
nesterov->setMomentum( 0.1f );
Trainer *trainer = nesterov;

Annealer *annealer = new Annealer( cl );
annealer->setLearningRate( 0.002f );
annealer->setAnneal( 0.97f );
Trainer *trainer = annealer;
```

## Train

eg:
```c++
NetLearner netLearner(
    trainer, net,
    Ntrain, trainData, trainLabels,
    Ntest, testData, testLabels );
netLearner.setSchedule( numEpochs );
netLearner.setBatchSize( batchSize );
netLearner.learn();
// learning is now done :-)
```

## Test

eg
```c++
// (create a net, as above)
// (train it, as above)
// test, eg:
BatchLearner batchLearner( net );
int testNumRight = batchLearner.test( batchSize, Ntest, testData, testLabels );
```

## Weight initialization

* By default an `OriginalInitializer` object is used to initialize weights (a bit hacky, but changing this would need a major version bump)
* You can create an instance of `UniformInitializer`, and assign this to the ConvolutionalMaker by doing for example `->setWeightInitializer( new UniformInitializer(1.0f) )`, to use a uniform initializer
  * uniform initializer assigns weights sampled uniformally from the range +/- ( initialWeights divided by fanin)
* possible to create other WeightsInitializers if we ant

## More details

You can find more details in the Doxygen-generated docs at [doxy docs for 4.x.x](http://deepcl.hughperkins.com/4.x.x/html/annotated.html)

