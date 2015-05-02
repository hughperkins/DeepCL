# What's done / what's planned

## Changes in next 4.x.x

* There are some changes and deprecations in the future 4.x.x   Please go to [Changes in 4.x.x](https://github.com/hughperkins/DeepCL/blob/4.x.x/doc/Changes.md) to check these, and raise an issues for any changes or deprecations you're not 100% comfortable with.

## To do

* To do basically moved to 'issues' now.  Please check in [issues](https://github.com/hughperkins/DeepCL/issues) for ideas for
  * things you might consider working on yourself, or
  * to add things you want others (eg me :-) ) to work on

## 5.x.x changes, compared to 4.x.x:

### New:

* Added Annealer trainer, and 'anneal=' commandline option
  * python and lua wrappers can also create Annealer trainer, as well as the existing SGD trainer

### Changes:

* bunch of changes to many of the classes now in `src/batch` directory, ie xxxBatcher classes, xxxNetAction type classes, and NetLearnerxxx classes, hence bumped major version to `5`, eg
  * XXXBatcher.tick: added parameter 'epoch' to 'tick' and 'run' methods
  * OnDemandBatcher constructor takes in a Batcher* object
  * created new Batcher2, and NetAction2 classes
  * removed BatchLearner class

## 4.x.x changes, compared to 3.x.x:

## New

* lua wrappers created, using swig
* added new ActivationLayer layer type, to implement relu, sigmoid etc in a separate layer from the convolutional,
and fc layers
* in commandline netdef, can put 'z' at the end of a convolutional layer to make it zero-padded, eg `32c5z`
* added dropout:
  * in commandline netdef, add `-drop`, eg like: `8c5z-drop-relu`
  * in C++, use a DropoutMaker, like `net->addLayer( DropoutMaker::instance()->dropRatio(0.5f) );`
* added momentum, via new [SGD](src/SGD.h) object, which can be passed to anywhere that accepts a [Trainer](src/Trainer.h) object.
* added new commandline parameter `gpuindex`, to choose which gpu device to target (thank-you Josef Moudrik for this addition)

### Changes

* lua module changes name from 'luaDeepCL' to 'LuaDeepCL'
* lua build method changes from build.sh to cmake
* default for convolutional and fc layers in commandline is now linear, instead of relu and tanh respectively
  * true for netdef syntax
  * true also for c++ Maker classes
  * will be true also for Python, Lua (not sure if it is true at the moment or not, would need to test)
* internal C++ name changes, somewhat in line with torch nomenclature, which seems simple, concise, easy to understand:
  * errors -> gradOutput
  * errorsForUpstream -> gradInput
  * resultsFromUpstream -> input
  * results -> output
  * dLossDOutput -> gradOutput
  * biasWeights -> bias
  * errorWeights -> gradWeights
* Have to create an OpenCLHelper object yourself, and pass it into NeuralNet constructor
  * not sure this increases the difficulty of usage too much?  and simplifies internal design quite a lot, since
one can assume that there is one and only one cl, shared everywhere (cf, otherwise MultiNet complicates things, or have to start reference-counting and stuff)
  * also increases flexibility slightly, can choose different gpu devices etc
* main header changes from [NeuralNet.h](NeuralNet.h) to [DeepCL.h](DeepCL.h), to try to reduce the whole-world-rebuilds effect during dev compilations
* need to pass an [SGD](src/SGD.h) object into NetLearner, QLearner, et al, instead of a learning rate
  * the SGD object holds the learningRate, and the (new) momentum parameter
* renamed OpenCLHelper project from OpenCLHelper to EasyCL, since easier to type, and remember
* prototyping build options have moved to `advanced` section in CMake options

### Deprecated

* templates removed; everything is `float` now, no `unsigned char`, or `T`
  * basically, templates are not very
    supported by scripting languages, complicating writing wrappers
* removed callbacks from BatchLearner, NetLearner, NetLearnerOnDemand, BatchLearnerOnDemand
  * replaced by simply calling `tickEpoch` or `tickBatch`, and doing what you want after each call, ie no longer actually need the callbacks
  * again, more compatible with writing wrappers for scripting languages, since inter-language callbacks tend to be hard work to implement
* idx-to-mat removed, since GenericLoader can directly handle reading mnist format now
* clconvolve1 executable removed (replaced by deeplclrun)
* activation layers within convolutional and fc layers removed

