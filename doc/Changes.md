# What's done / what's planned

## To do

* To do basically moved to 'issues' now.  Please check in [issues](https://github.com/hughperkins/DeepCL/issues) for ideas for
  * things you might consider working on yourself, or
  * to add things you want others (eg me :-) ) to work on

## New, in next version, 4.x.x

* lua wrappers created, using swig
* new set of python wrappers, in `python_swig` directory, using swig instead of cython
  * plausibly will be published to pypi as 'PyDeepCLSwig', in parallel to the existing Cython version

## Changes, in next version, 4.x.x

* lua module changes name from 'luaDeepCL' to 'LuaDeepCL'
* lua build method changes from build.sh to cmake

## Deprecated, in next version, 4.x.x

* templates removed; everything is `float` now, no `unsigned char`, or `T`
  * basically, templates are not very
    supported by scripting languages, complicating writing wrappers
* removing callbacks from BatchLearner, NetLearner, NetLearnerOnDemand, BatchLearnerOnDemand
  * replaced by simply calling `tickEpoch` or `tickBatch`, and doing what you want after each call, ie no longer actually need the callbacks
  * again, more compatible with writing wrappers for scripting languages, since inter-language callbacks tend to be hard work to implement
* idx-to-mat removed; since GenericLoader can directly handle reading mnist format now
* clconvolve1 executable removed (replaced by deeplclrun)

## Recent changes, to 4.x.x branch

* 25th April: added independent activation layers, and changed default for convolutional and fc layers to linear
* 25th April: cleaned up the benchmarks a lot, added them to jenkins, added a couple more, created an Angular/Bootstrap page to display them [DeepCL benchmarks](http://hughperkins.github.io/DeepCL/benchmarking/)
* 21st April: Added lua wrappers to luarocks repository
* 18th April: with `loadweights=1`, will load old weights file format too, not just refuse to load
* 18th April: with `loadweights=1`, if the file doesnt match current options, you can now override if you want, at the risk of crashing, or loading inappropriate weights

