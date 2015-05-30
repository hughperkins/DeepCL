# Recent changes

Recent changes can be viewed by looking at [Releases](https://github.com/hughperkins/DeepCL/releases)

# Stable public API

How to determine the current stable public API is documented at [Stable API](PublicApis.md)

# Change in next release

## New in next release

* `deepclrun` becomes `train`: handles training, and validation, using labelled data
* `deepclexec` becomes `predict`: handles creating predictions from unlabelled data
* `predict` can take input from a file, using GenericLoader, same formats as training, or from stdin
* `predict` can output to a file, or to stdout, in text or binary format

## Changes in next release

* normalization layer translate and scale are now stored in weights file, whose version has now been bumped to 3
  * weights files versions 1 and 3 can both be read, any updated weights file will be written as version 3

## Changes under the covers, in next release

* GenericLoader::load can take a value of `0` for labels, which means labels wont be loaded
  * used by `predict`

