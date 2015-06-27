# Recent changes

Recent changes can be viewed by looking at [Releases](https://github.com/hughperkins/DeepCL/releases)

# Stable public API

How to determine the current stable public API is documented at [Stable API](PublicApis.md)

# Change in next release

## New in next release


## Changes in next release

* `deepclrun` becomes `deepcl_train`: handles training, and validation, using labelled data
* `deepclexec` becomes `deepcl_predict`: handles creating predictions from unlabelled data
* lua wrappers removed.  Effort for lua has moved to [cltorch](https://github.com/hughperkins/cltorch) and [clnn](https://github.com/hughperkins/clnn), both of which are well underway :-)
* `deepcl_predict` can take input from a file, using GenericLoader, same formats as training, or from stdin
* `deepcl_predict` can output to a file, or to stdout, in text or binary format

## Changes under the covers, in next release

* migrated the underlying maths functions to use the new lua kernel templater

