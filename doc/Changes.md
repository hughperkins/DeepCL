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

## Done

  * forward/backward propagation, for convolutional networks, using OpenCL
  * square loss
  * zero-padding
  * relu activation
  * tanh activation
  * linear activation
  * some optimization of the OpenCL kernels
  * can save/load weights
  * can use 'fluent' style to setup the networks
  * unit-tests for forward propagation
  * numerical validation for backward propagation
  * softmax activation function
  * cross entropy loss
  * multinomial cross entropy loss
  * get working with [kgs go data](https://github.com/hughperkins/kgsgo-dataset-preprocessor)
    * created GenericLoader, which automatically detects filetype
    * created Kgsv2Loader, which handles kgsgo v2 data files
    * added loadondemand, so can load data as we go, during learning, rather than trying to load the entire dataset in one go
  * create a 'transforming input' layer, to handle things like:
    * conversion from `unsigned char *` to `float *`
    * translation and scaling by mean and standard deviation
  * MCDNN
  * randomly translating input layer
  * Python bindings =>  Done (though could be improved of course...)
  * Q-learning Done (though could be improved of course)
  * generalization to larger images => kind of done, ish, for NORB
  * max-pooling
  * read network from a config file => soft of done with the `netdef` syntax
  * write a LuaJIT wrapper since Yann LeCun mentioned LuaJIT in his [AMA](http://www.reddit.com/r/MachineLearning/comments/25lnbt/ama_yann_lecun/) , ie at [http://www.reddit.com/r/MachineLearning/comments/25lnbt/ama_yann_lecun/chiyqzw](http://www.reddit.com/r/MachineLearning/comments/25lnbt/ama_yann_lecun/chiyqzw)
  * write the weights to file more often than once an epoch, so each time my machine goes down, after 1 day and 23 hours, I dont lose 2 days of learning :-P => added writeweightsinterval option

## Recent changes, to 4.x.x branch

* 21st April: Added lua wrappers to luarocks repository
* 18th April: with `loadweights=1`, will load old weights file format too, not just refuse to load
* 18th April: with `loadweights=1`, if the file doesnt match current options, you can now override if you want, at the risk of crashing, or loading inappropriate weights

