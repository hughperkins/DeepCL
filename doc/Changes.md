# Recent changes

Recent changes can be viewed by looking at [Releases](https://github.com/hughperkins/DeepCL/releases)

# Stable public API

How to determine the current stable public API is documented at [Stable API](PublicApis.md)

# Changes in next release

## Under the covers

* factorized applying bias into separate class for forward3 (it was already in a separate OpenCLKernel), and away from the convolutional forward opencl in forward4
* fixed a bug in forward4, where images larger than the square root of the maximum gpu workgroupsize sometimes has incorrect values for the last few pixels
* migrated to latest version of EasyCL, which handles storing the device dirty flag for us, rather than having lots of flags in our code like `weightsCopiedToHost`, and so on

