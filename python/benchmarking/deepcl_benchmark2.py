#!/usr/bin/python

# This was originally created to target inclusion in soumith's benchmarks as
# https://github.com/soumith/convnet-benchmarks
# extending this to handle also a couple of clarke and storkey type layers
# and some plausible mnist layers

from __future__ import print_function
import os
import sys
import time
import numpy as np
import random
import json
import subprocess
import PyDeepCL

cmd_line = 'cd python; python setup.py build_ext -i; PYTHONPATH=. python'
for arg in sys.argv:
    cmd_line += ' ' + arg
print('cmd_line: [' + cmd_line + ']')
    # benchmarking/deepcl_benchmark2.py'
num_epochs = 10
batch_size = 128  # always use this, seems pretty standard
runs = [
    # format for single layer: ('[label]','[inputplanes]i[inputsize]-[numfilters]c[filtersize]', 'layer')
    # format for full net: ('[label]', '[netdef]', 'fullnet')
    ('soumith1', '3i128-96c11', 'layer'),  
    ('soumith2', '64i64-128c9', 'layer'),
    ('soumith3', '128i32-128c9', 'layer'),
    ('soumith4', '128i16-128c7', 'layer'),
    ('soumith5', '384i13-384c3', 'layer'),
    ('maddison-convolve', '128i19-128c3', 'layer'),
    ('maddison-fc', '128i19-361n', 'layer'),
    ('mnist-c1', '1i28-8c5', 'layer'),
    ('mnist-c2', '8i14-16c5', 'layer'),
    ('mnist-fc', '16i7-150n', 'layer'),
    ('mnist-full', '1i24-8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-10n', 'fullnet'),
    ('mnist-full-factorized', '1i24-8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-10n', 'fullnet'),
    ('maddison-full', '8i19-12*128c3z-relu-361n', 'fullnet'),
    ('maddison-full-factorized', '8i19-12*(128c3z-relu)-361n', 'fullnet')
]

def write_results( label, net_string, layer, benchmark_type, direction, time_ms ):
    global cmd_line
    results_dict = {}
    results_dict['label'] = label
    results_dict['type'] = benchmark_type
    results_dict['format'] = 'v0.4'
    results_dict['direction'] = direction
    results_dict['net_string'] = net_string
    if layer is not None:
        results_dict['layer_string'] = layer.asString()
    results_dict['time_ms'] = str(time_ms)
    results_dict['cmd_line'] = cmd_line

    f = open('results.txt', 'a')
    json.dump(results_dict, f)
    f.write( '\n' )
    f.close()

def time_layer(num_epochs, label, batch_size, net_string):
    print('building network...')
    input_string, layer_string = net_string.split('-')
    input_planes, input_size = map(lambda x: int(x), input_string.split('i'))
    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet( cl, input_planes, input_size )
    net.addLayer( PyDeepCL.ForceBackpropMaker() ) # this forces the next layer to backward gradients to
                          # this layer
    print( net.asString() )
    if 'c' in layer_string:
        num_filters, filter_size = map(lambda x: int(x), layer_string.split('c'))
        net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(num_filters)
            .filterSize(filter_size).biased() )
    elif 'n' in layer_string:
        num_neurons = int(layer_string.split('n')[0])
        net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(num_neurons).imageSize(1).biased() )
    else:
        raise Exception('layer_string {layer_string} not recognized'.format(
            layer_string=layer_string))
    print( net.asString() )
    net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(1).imageSize(1) )
    net.addLayer( PyDeepCL.SoftMaxMaker() )
    print( net.asString() )

    images = np.zeros((batch_size, input_planes, input_size, input_size), dtype=np.float32)
    images[:] = np.random.uniform(-0.5, 0.5, images.shape)
    labels = np.zeros((batch_size,), dtype=np.int32)
    
    print('warming up...')
    #try:
    net.setBatchSize(batch_size)

    # warm up forward
    for i in range(9):
        last = time.time()
        net.forward( images )
        now = time.time()
        print('  warm up forward all-layer time', ( now - last ) * 1000, 'ms' )
        last = now
        net.backwardFromLabels(labels)
        now = time.time()
        print('   warm up backward all-layer time', (now - last) * 1000, 'ms' )
        last = now

    layer = net.getLayer(2)
    print('running forward prop timings:')
    for i in range(num_epochs):
        layer.forward()
    now = time.time()
    print('forward layer total time', ( now - last) * 1000, 'ms' )
    print('forward layer average time', ( now - last ) * 1000 / float(num_epochs), 'ms' )
    # forward_time_per_layer_ms = ( now - last ) / float(num_epochs) * 1000
    # writeResults( label + ', ' + net_string + ', ' + layer.asString() + ', forward=' + str( ( now - last ) / float(num_epochs) * 1000 ) + 'ms' )
    write_results( label=label, net_string=net_string, layer=layer, direction='forward',
        benchmark_type='layer', time_ms=( now - last ) / float(num_epochs) * 1000 )

    print('warm up backwards again')
    layer.backward()
    layer.backward()
    print('warm up backwards done. start timings:')

    now = time.time()
    last = now
    for i in range(num_epochs):
        layer.backward()
    now = time.time()
    print('backward layer total time', (now - last)*1000, 'ms' )
    print('backward layer average time', ( now - last ) * 1000 / float(num_epochs), 'ms' )
    # writeResults( label + ', ' + net_string + ', ' + layer.asString() + ', backward=' + str( ( now - last ) / float(num_epochs) * 1000 ) + 'ms' )
    write_results( label=label, net_string=net_string, layer=layer, 
        direction='backward', benchmark_type='layer', time_ms=( now - last ) / float(num_epochs) * 1000 )
    last = now

def time_fullnet(num_epochs, label, batch_size, net_string):
    print('building network...')
    split_net_string = net_string.split('-')
    input_string = split_net_string[0]
    netdef = '-'.join(split_net_string[1:])
    input_planes, input_size = map(lambda x: int(x), input_string.split('i'))
    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet(cl, input_planes, input_size )
    PyDeepCL.NetdefToNet.createNetFromNetdef(net, netdef)
    print( net.asString() )


    images = np.zeros((batch_size, input_planes, input_size, input_size), dtype=np.float32)
    images[:] = np.random.uniform(-0.5, 0.5, images.shape)
    labels = np.zeros((batch_size,), dtype=np.int32)

    print('warming up...')
    #try:
    net.setBatchSize(batch_size)

    # warm up forward
    for i in range(8):
        last = time.time()
        net.forward( images )
        now = time.time()
        print('  warm up forward all-layer time', (now - last)*1000.0, 'ms')
        last = now

    print('warming up backward:')
    last = time.time()
    net.backwardFromLabels(labels)
    now = time.time()
    print('   warm up backward time', (now - last) * 1000, 'ms' )
    last = now
    net.backwardFromLabels(labels)
    now = time.time()
    print('   warm up backward time', (now - last) * 1000, 'ms' )

    total_forward = 0
    total_backward = 0
    last = time.time()
    num_epochs = 0
    while total_forward < 1000 or total_backward < 0: # make sure collect suffiicnet timing
#    for epoch in range(num_epochs):
        print('epoch {epoch}'.format(epoch=num_epochs+1))
        print('run forward for real...')
        # last = time.time()
        net.forward(images)
        now = time.time()
        diff = now - last
        forward_ms = diff * 1000.0
        total_forward += forward_ms
        print('forward time: {forward_ms}ms'.format(
            forward_ms=forward_ms))
        last = now

        print('backward for real:')
        # last = time.time()
        net.backwardFromLabels(labels)
        now = time.time()
        diff = now - last
        backward_ms = diff * 1000.0
        total_backward += backward_ms
        print('backward time: {backward_ms}ms'.format(
            backward_ms=backward_ms))
        last = now
        num_epochs += 1

    print('num_epochs: {num_epochs}'.format(num_epochs=num_epochs))
    average_forward = total_forward / num_epochs
    average_backward = total_backward / num_epochs
    print('average forward time: {forward_ms}ms'.format(
        forward_ms=average_forward))
    print('average backward time: {backward_ms}ms'.format(
        backward_ms=average_backward))

    write_results( label=label, net_string=net_string, layer=None,
        benchmark_type='fullnet', direction='forward', 
        time_ms=average_forward )
    write_results( label=label, net_string=net_string, layer=None,
        benchmark_type='fullnet', direction='backward', 
        time_ms=average_backward )

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
    for (label, net_string, benchmark_type) in runs:
        print( '' )
        print( 'CONFIG: ', label, net_string)

        if benchmark_type == 'layer':
            time_layer(num_epochs, label=label, batch_size=batch_size, net_string=net_string)
        elif benchmark_type == 'fullnet':
            time_fullnet(num_epochs, label=label, batch_size=batch_size, net_string=net_string)
        else:
            raise Exception('unrecognized benchmark type [' + benchmark_type + '], can choose "layer" or "fullnet"')

if __name__ == '__main__':
    """
    if just one run is chosen, we run that, and exit
    otherwise, if no runs are chosen, or some runs are chosen,
    then we will spawn one process per run name
    """
    if len(sys.argv) == 2:
        # one run chosen, so run it, directly
        chosen_runs = []
        for chosen_label in sys.argv[1:]:
            for label, run_string, benchmark_type in runs:
                if label == chosen_label:
                    chosen_runs.append((label, run_string, benchmark_type))
        assert len(chosen_runs) == 1
        go(chosen_runs)
    else:
        # get all selected tests
        # then run each one in separate process
        chosen_labels = sys.argv[1:]
        if len(chosen_labels) == 0:
            for run in runs:
                (label, _, _) = run
                chosen_labels.append(label)
        # now run each one
        for label in chosen_labels:
            print('spawning for {label}'.format(
                label=label))
            print('sys.argv[0]', sys.argv[0])
            p = subprocess.Popen(
                ['python', sys.argv[0], label])
            p.wait()
#            while p.poll() is None:
#                output = p.stdout.readline()
#                print(output)
            print('done for {label}'.format(label=label))
        print('all done :-)')

