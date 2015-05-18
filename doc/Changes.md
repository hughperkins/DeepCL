# Recent changes

Recent changes can be viewed by looking at [Releases](https://github.com/hughperkins/DeepCL/releases)

# Stable public API

How to determine the current stable public API is documented at [Stable API](PublicApis.md)

# Next release

## New in next release

* Josef Moudrik has created `deepclexec`, to run prediction, on pre-trained weights, for new data samples
* Added jpeg loader, so can load imagenet data now

## Changes in next release

* Added build dependency on libjpeg 6.2, or compatible (eg libjpeg-turbo), in order to add the jpeg loader

## Changes under the covers, in next release

* factorized applying bias into separate class for forward3 (it was already in a separate OpenCLKernel), and away from the convolutional forward opencl in forward4, and forward1
* fixed a bug in forward4, where images larger than the square root of the maximum gpu workgroupsize sometimes has incorrect values for the last few pixels
* migrated to latest version of EasyCL, which handles storing the device dirty flag for us, rather than having lots of flags in our code like `weightsCopiedToHost`, and so on
* GenericLoaderv2 is now stateful, rather than using static methods as per original GenericLoader
  * new NetLearnerOnDemandv2 uses GenericLoaderv2, as does new OnDemandBatcherv2
  * deepclrun migrated to use GenericLoaderv2
  * GenericLoaderv1Wrapper wraps existing GenericLoader implementations, so no need to re-write those in any way for now, and any new GenericLoader implementations can continue to be v1, via the wrapper, if they dont need state
  * making GenericLoaderv2 stateful means we can read a jpeg manifest, eg for imagenet et al, once, and then hold it in memory

