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
* 21st March:
  * global rename 'board' and 'Board' to 'image' and 'Image', to make more generic
* 17th February:
  * migrated max-pooling backprop to gpu
* 15th February:
  * removed runtime dependency on *.cl files
* 13th February:
  * created PropagateAuto, which tries each kernel once, for one batch, and then picks the fastest
* 10th February:
  * increased speed of fully-connected propagation by 5-8 times :-)
* 7th Februrary:
  * added support for kgs go v2 format [https://github.com/hughperkins/kgsgo-dataset-preprocessor](https://github.com/hughperkins/kgsgo-dataset-preprocessor)
  * added loadondemand, which makes it possible to read a file in chunks, as learning proceeds, reducing memory requirements
  * created `GenericLoader.cpp`, which can auto-detect file types, given the path of the images file
* 5th February:
  * added multinet: multi-column networks
* 3rd Februrary:
  * added RandomTranslations layer
* 2nd February:
  * added RandomPatches layer
* 1st Februrary:
  * builds and runs on Windows
* 31st January:
  * added NormalizationLayer
  * templated InputLayer, for different input array types, which simplifies BatchLearner class
* 30th January:
  * add scaled tanh, ie 1.7159f * tanh(0.66667f * x )
* 29th January:
  * fix showstopper bug in idx-to-mat
  * add {sigmoid}, {tanh}, {scaledtanh}, {linear} options
  * add normalization=maxmin, normalization=stddev (default)
  * add {padzeros} options
* 27th January:
  * builds on Windows
* 26th January:
  * Unified mnist and norb testing executable
  * Implemented network-definition, as specified in [Ciresan et al](arxiv.org/pdf/1202.2745.pdf)
  * Created idx-to-mat, to convert from mnist idx format to norb mat format
  * Massively overhauled this readme in the light of these changes
  * Added annealed learning rate, as described in [Ciresan et al](http://ijcai.org/papers11/Papers/IJCAI11-210.pdf)
  * Add reasonably robust restart-file
* 25th January:
  * Added gpu implementation for max-pooling forward-prop
  * Added padZeros option for max-pooling
  * Accelerated backpropweights kernel a bit, for larger images (ie norb, 96x96)
* 24th January:
  * added max-pooling layer (albeit in cpu for now)
  * created draft 'lenet5' implementation, but it's not quite the same, specifically:
    * lenet-5 has RBF layers at the end
    * lenet-5 has multiple of these RBF and fully-connected layers at the end
    * lenet-5 is not using max-pooling but something more like average-pooling, and it has an activation function applied (sigmoid)
  * added the mnist training config from [convnetjs](http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)
  * noticed we are actually in January, not December, and updated the name of the month in this section appropriately :-P
* 23rd January:
  * created `testmnist-mpi`, to experiment with using mpi to parallelize across multiple compute nodes (which must each have a GPU, which GPUs must ideally each be the same model/specifications)
* 22nd January:
  * re-added FullyConnectedLayer, which is now a wrapper around ConvolutionalLayer, with one filter per output node.  So, if we want a 28x28 image as the output, this will need 784 filters in the underlying convolutional layer, which
sounds excessive, but this is how a fully connected layer works :-)
  * best mnist accuracy now at 98.6%
* 21st January:
  * added softmax layer, for per-column configuration, ie multi-planar output, with imagesize 1
    * tested once on mnist: 97.65% test accuracy after 12 epochs; 98.09% after 20 epochs
* week up to 21st January: 
  * added sigmoid activation
  * added cross-entropy loss layer
  * migrated to recurse on dLoss/dSum, rather than dLoss/dOutput, ie on partial derivative of loss with input to activation function for each neuron, rather than with output.  Recursing on input instead of output is faster
  * changed learning rate, so that the square of the sum of the weight changes equals approximately the change in loss, for smallish changes in w, so that we can numerically validate easily
  * validated backpropagation numerically
  * migrated to use explicit square-loss layer
  * moved sources to `src` sub-directory, so root directory cleaner
  * created `SLOW_` prefix for slow tests, so can run with `gtest_filter=-SLOW*` to ignore slow tests


