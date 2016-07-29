#!/usr/bin/python

# this is for experimenting prior to incorporation in deepcl_benchmark.py
#
# you might need to add PYTHONPATH=. to run it, like:
#
#     PYTHONPATH=. python benchmarking/test_singlelayer.py

from __future__ import print_function

import deepcl_benchmark

def go():
    inputPlanes = 3
    inputSize = 96
    filterSize = 5
    outputPlanes = 8 # this means: number of filters
    batchSize = 128
    deepcl_benchmark.time_layer(3, batchSize, inputPlanes, inputSize, outputPlanes, filterSize)

if __name__ == '__main__':
    go()

