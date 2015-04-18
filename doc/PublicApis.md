# Public APIs

## Semantic versioning

DeepCL aims to follow [semantic versioning](semver.org).  This page documents which parts of the DeepCL API
will hopefully be stable in between major version changes.

## Stable APIs

The following APIs will aim to only be modified in major version changes:
* commandline parameters and options for `deepclrun`
* Cython Python wrappers API: all existing wrapped classes and methods, except for q-learning
* The methods and classnames of the following C++ classes, not including properties/attributes:
  * NeuralNet class, except for the following methods:
    * cloneLossLayerMaker
    * epochMaker
    * printBiasWeightsAsCode
    * printWeightsAsCode
    * maker
    * getCl
    * clone
    * printOutput
    * printWeights
  * All xxxMaker classes and class methods
  * Following methods from GenericLoader class:
    * `void getDimensions( std::string trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize );`
    * `STATIC void load( std::string imagesFilePath, float *images, int *labels, int startN, int numExamples );`
  * NetDeftoNet.createNetFromNetdef method
  * WeightsPersister class, except for `copyArray` method
  * NetLearner class, except for `postEpochTesting` method
  * Batcher class class methods
  * OnDemandBatcher class methods
  * NetLearnerOnDemand class methods, except for `postEpochTesting` method
* Compiler standards should ideally be covered, to the extent that the ability to compile on linux
using g++ with only `-std=c++0x`, and on Windows, using Visual Studio 2010 Express should not be removed without a major version change

This is not to say that new methods cant be added in between major versions.  They can, and will be,
but ideally these existing methods wont be modified, at least: not intentionally.

## Unstable APIs

The following APIs are currently unstable, and can be modified in between major version changes:
* Lua API
  * eg, might change to use Torch arrays for input, plausibly
* Swig Python wrappers API
  * eg, might change to use numpy arrays for input, plausibly
* The C++ q-learning module and classes
* All other C++ classes and methods not stated explicitly in 'Stable APIs' section above
* All attributes and properties on C++ classes
* Build process is not included
* Packaging process, eg luarocks, python eggs etc, is not included
* Names of installable packages, eg name of project on pypi, luarocks etc, is not included

## Changes to this document

* Classes and methods can freely be added to this document in between major version changes
  * obviously that doesnt mean you can just add everything to this document, and submit a pull request ;-)  If you
need something added to this document, please let me know, eg via email, raising a Github issue, or through
the mailing list
* Classes and methods can ideally only be removed from this document on a major version change

