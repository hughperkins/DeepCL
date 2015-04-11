<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [What's done / what's planned](#whats-done--whats-planned)
- [Recent changes](#recent-changes)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

What's done / what's planned
============================

* Done:
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
* Planned, short-term:
  * Reading papers right now...
  * Currently, I'm interested in:
    * the Atari paper
    * LTSM
    * So, anything which furthers being able to pursue either of these is likely to be sooner rather than later
    * Specifically, these need things like:
      * Q-learning (for Atari)
      * probably more generalized network, maybe even more general than a DAG even, for LTSM
  * I'm also tempted to write a LuaJIT wrapper since:
    * sounds easy, close to my skill-set, I've done lua before
    * Yann LeCun spoke about LuaJIT in his [AMA](http://www.reddit.com/r/MachineLearning/comments/25lnbt/ama_yann_lecun/)
* Plausible, medium-term (pull requests welcome):
  * drop-out ... pretty important :-)
  * scaling? rotations? mirroring?
  * testing result averaged over several propagations (used in conjunction with `rp`)
  * sparse connectivity between feature maps in adjacent layers
  * ~~skip~~ stride, (skip is described in [Ciresan et al, 2011](http://arxiv.org/pdf/1102.0183v1.pdf) , and stride is a similar, but plausibly more standard concept? )
  * symmetric filters
  * fuse convolutional and max-pooling layers, so can optimize more
  * maybe L2 regularization?
  * generalization to non-square images
  * more general DAGs?
* Maybe sometime, possibly:
  * mpi so can run over several gpus, spread across multiple hosts???
    * implemented mpi in `testmnist-mpi`.  If works ok, will generalize to something more permanent => since it didnt seem obvious how to use it, ie you have to divide the learningrate by the number of nodes, I never use this at the moment
* Rejected, for now:
  * migrate to use `async_work_group_copy`? => rejected, seems it's actually slower, in my experiments, at least on nvidia?
  * [DropConnect](http://cs.nyu.edu/~wanli/dropc/dropc.pdf) => Rejected, since, per [Sandle Dieleman's solution to the Galaxy Zoo challenge](http://benanne.github.io/2014/04/05/galaxy-zoo.html), seems like dropconnect is slower and doesnt convincingly add value, compared to dropout

Recent changes
==============

Dates are dates of code change / commit, rather than date merged into master, or tagged.
* ( cleaned this for now :-) )

