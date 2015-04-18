# Public APIs

## Semantic versioning

DeepCL aims to follow [semantic versioning](semver.org).  This page documents which parts of the DeepCL API
will hopefully be stable in between major version changes.

## Stable APIs

The following APIs will aim to only be modified in major version changes:
* commandline parameters and options for `deepclrun`
* Cython Python wrappers API: all existing wrapped classes and methods, except for q-learning
* Any C++ methods marked as `PUBLICAPI`.  This is a null macro, that does nothing.  Its sole job
is to mark methods that should be stable within major versions
  * their containing class should also not be removed or change name
* The following classes contain methods tagged as `PUBLICAPI`, and therefore should not be removed, or change name, except in major version changes:
  * NeuralNet
  * Most xxxMaker classes (but not all)
  * NetdefToNet
  * GenericLoader
  * WeightsPersister
  * Batcher
  * OnDemandBatcher
  * NetLearner
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
* NetLearnerOnDemand c++ class might be merged into NetLearner class plausibly
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

