#!/usr/bin/python

# This was originally created to target inclusion in soumith's benchmarks as
# https://github.com/soumith/convnet-benchmarks
# extending this to handle also a couple of clarke and storkey type layers
# and some plausible mnist layers

from __future__ import print_function

import os
import sys
import time
import array
import random
import PyDeepCL

num_epochs = 10
batch_size = 128  # always use this, seems pretty standard
runs = [
    ('soumith1', '3i128-96c11'),  # format: ('[label]','[inputplanes]i[inputsize]-[numfilters]c[filtersize]')
    ('soumith2', '64i64-128c9'),
    ('soumith3', '128i32-128c9'),
    ('soumith4', '128i16-128c7'),
    ('soumith5', '384i13-384c3'),
    ('maddison-convolve', '128i19-128c3'),
    ('maddison-fc', '128i19-361n'), # this crashes currently, not sure why, since it's just allocated 63MB
    # memory for weights, so it's probably a bug in my code, rather than a 
    # theoretical limitation of current implementation
    ('mnist-c1', '1i28-8c5'),
    ('mnist-c2', '8i14-16c5'),
    ('mnist-fc', '16i7-150n')
]

def writeResults( resultsLine ):
    f = open('results.txt', 'a')
    f.write( resultsLine + '\n' )
    f.close()

def time_layer(num_epochs, label, batch_size, net_string):
    print('building network...')
    input_string, layer_string = net_string.split('-')
    input_planes, input_size = map(lambda x: int(x), input_string.split('i'))
    net = PyDeepCL.NeuralNet( input_planes, input_size )
    net.addLayer( PyDeepCL.ForceBackpropMaker() ) # this forces the next layer to backprop gradients to
                          # this layer
    print( net.asString() )
    if 'c' in layer_string:
        num_filters, filter_size = map(lambda x: int(x), layer_string.split('c'))
        net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(num_filters)
            .filterSize(filter_size).biased().linear() )
    elif 'n' in layer_string:
        num_neurons = int(layer_string.split('n')[0])
        net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(num_neurons).imageSize(1).biased().linear() )
    else:
        raise Exception('layer_string {layer_string} not recognized'.format(
            layer_string=layer_string))
    print( net.asString() )
    net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(1).imageSize(1) )
    net.addLayer( PyDeepCL.SoftMaxMaker() )
    print( net.asString() )

    images = array.array( 'f', [0] * (batch_size*input_planes*input_size*input_size) )
    for i in range( batch_size*input_planes*input_size*input_size ):
        images[i] = random.random() - 0.5
#    grad = array.array('f',[0] * batch_size * outputPlanes * (input_size - filterSize + 1) )
#    for i in range( batch_size * outputPlanes * (input_size - filterSize + 1) ):
#        grad[i] = random.random() - 0.5
    labels = array.array('i',[0] * batch_size )
    
    print('warming up...')
    #try:
    net.setBatchSize(batch_size)

    # warm up forward
    for i in range(8):
        last = time.time()
        net.propagate( images )
        now = time.time()
        print('  warm up propagate all-layer time', now - last )
        last = now
    net.backPropFromLabels( 0.001, labels )
    now = time.time()
    print('   warm up backprop all-layer time', now - last )
    last = now

    layer = net.getLayer(2)
    print('running forward prop timings:')
    for i in range(num_epochs):
        layer.propagate()
    now = time.time()
    print('forward layer total time', now - last )
    print('forward layer average time', ( now - last ) / float(num_epochs) )
    writeResults( label + ', ' + net_string + ', ' + layer.asString() + ', forward=' + str( ( now - last ) / float(num_epochs) * 1000 ) + 'ms' )

    print('warm up backwards again')
    layer.backProp(0.001)
    layer.backProp(0.001)
    print('warm up backwards done. start timings:')

    now = time.time()
    last = now
    for i in range(num_epochs):
        layer.backProp(0.001)
    now = time.time()
    print('backwar layer total time', now - last )
    print('backwar layer average time', ( now - last ) / float(num_epochs) )
    writeResults( label + ', ' + net_string + ', ' + layer.asString() + ', backward=' + str( ( now - last ) / float(num_epochs) * 1000 ) + 'ms' )
    last = now

def time_run(fn):
    times = []
    fn()  # warm-up call, outputPlanest timed
    for _ in range(repeat):
        start = time.time()
        for _ in range(number):
            fn()
        times.append((time.time() - start) / number)
    return min(times)

#def parse_custom_config(s):
#    # parses a custom configuration string of the format:
#    # AxB-CcD where A: input channels, B: input size,
#    # C: output channels, D: kernel size, E: batchsize
#    run = {'batch_size': 128 }
#    defs = {'i': ['input_planes', 'input_size'],
#            'k': ['outputPlanes', 'filterSize'],
#            'b': ['batch_size'] }
#    for part in s.split(','):
#        p, args = part[0], map(int, part[1:].split('x'))
#        run.update(zip(defs[p], args))
#    return run

#def parse_run_string(run_string):
#    split_string = run_string.split('-')
#    input_string = split_string[0]
#    net_string = split_string[1]
#    split_input = input_string.split('x')
#    
#    split_conv = conv_string.split('c')
#    return {'input_planes': input_strin

def go(runs):
    global batch_size
    for (label, net_string) in runs:
        print( '' )
        print( 'CONFIG: ', label, net_string)

        time_layer(num_epochs, label=label, batch_size=batch_size, net_string=net_string)

if __name__ == '__main__':
    chosen_runs = runs
    if len(sys.argv) > 1:
        chosen_runs = []
        for chosen_label in sys.argv[1:]:
            for label, run_string in runs:
                if label == chosen_label:
                    chosen_runs.append((label, run_string))
        # allow specifying the runs on command line, 1-indexed (i.e., 1 2 5)
#        runs = [runs[int(r) - 1] for r in sys.arsgv[1:]]
        # allow specifying custom configurations on command line (e.g., i3x80x15,k32x3x7,b256)
#        runs.extend([parse_custom_config(r) for r in sys.argv[1:] if r[0] == 'i'])

    go(chosen_runs)

