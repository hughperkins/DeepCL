// I'm using (since 13th March 2015) convnetjs as a reference implementation, to double-check
// my calculations, since convnetjs:
// - is straightforward to read
// - widely used/forked, therefore probably correct
// - easy to run, doesnt need a gpu etc
// - I like nodejs :-)

// see README.md for more details

"use strict";

var fs = require('fs');
var convnetjs = require('convnetjs');
var PNG = require('pngjs').PNG;
var http = require('http');
var Random = require('random-js'); // we use random-js, so we can generate random numbers repeatably
                                   // and following identical sequence to mt19937 in c++
var mt = Random.engines.mt19937();

function downloadFile( url, targetFilepath, callback ) {
    var file = fs.createWriteStream( targetFilepath );
    http.get( url, function( response ) {
        response.pipe(file);
        file.on( 'finish', function() {
            file.close( callback( targetFilepath ) );
        });
    });
}

function doDownloads( callback ) {
    // download the data files, if not present
    var dataFiles = [];
    dataFiles.push( 'mnist_labels.js' );
    dataFiles.push( 'mnist_batch_0.png' );
    var urlBase = 'http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist';
    var targetDirectory = __dirname + '/data';
    if( !fs.existsSync(targetDirectory) ) {
        fs.mkdirSync( targetDirectory );
    }
    var numDownloading = 0;
    var numDownloaded = 0;
    for( var i in dataFiles ) {
        var filename = dataFiles[i];
        if( !fs.existsSync( targetDirectory + '/' + filename ) ) {
            console.log('downloading ' + urlBase + '/' + filename + ' ...');
            numDownloading++
            downloadFile( urlBase + '/' + filename, targetDirectory + '/' + filename, function() {
                console.log('downloaded ' + filename );
                numDownloaded++;
                if( numDownloaded == numDownloading ) {
                    console.log('finished downloading all files');
                    callback();
                }
            });
        }
    }
    if( numDownloading == 0 ) {
        callback();
    }
}

function openPng( path, callback ) {
    fs.createReadStream(path)
        .pipe(new PNG({
            filterType: 4
        }))
        .on('parsed', function() {
        console.log('parsed png');
        console.log('png width: ' + this.width);
        console.log('png height: ' + this.height);
        callback( this.data );
    });
}

// add layers at a low-level, so we can get very precise control
// over eg, the high-level version adds an extra fc, just beneath
// any softmax.
convnetjs.Net.prototype.addLayer = function( type, opt ) {
    if( typeof opt == 'undefined' ) {
        opt = {};
    }
    var prev = this.layers[ this.layers.length - 1 ];
    var newLayer;
    opt.in_sx = prev.out_sx;
    opt.in_sy = prev.out_sy;
    opt.in_depth = prev.out_depth;
    if( type == 'tanh' ) {
        newLayer = new convnetjs.TanhLayer(opt);
    } else if( type == 'relu' ) {
        newLayer = new convnetjs.ReluLayer(opt);
    } else if( type == 'sigmoid' ) {
        newLayer = new convnetjs.SigmoidLayer(opt);
    } else if( type == 'pool' ) {
        newLayer = new convnetjs.PoolLayer(opt);
    } else if( type == 'fc' ) {  
        if( typeof opt.num_neurons == 'undefined' ) {
            console.log('required option: num_neurons, not defined' );
        }
        newLayer = new convnetjs.FullyConnLayer( opt );
    } else if( type == 'softmax' ) {
        newLayer = new convnetjs.SoftmaxLayer( opt );
    } else if( type == 'conv' ) {
        newLayer = new convnetjs.ConvLayer( opt );
    } else {
        console.log('unknown type ' + type );  
    }
    this.layers.push( newLayer );
}

convnetjs.Net.prototype.print = function() {
    for( var i in this.layers ) {
        console.log( i + ' ' + this.layers[i].layer_type );
    }
}

convnetjs.Vol.prototype.get_n = function() {
    return this.sx*this.sy*this.depth;
}

function createNet() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:28, out_sy:28, out_depth:1});
    layer_defs.push({type:'softmax', num_classes:10});

    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    net.layers.splice(1,100); // remove original filters, so we can
                              // use low-level methods to add our own

    net.addLayer( 'pool', { 'sx':7, 'stride':7 });
    net.addLayer( 'conv', {'filters': 2, 'sx': 1, 'pad': 0 } );
    //net.addLayer( 'fc', {'num_neurons': 10} );
    //net.addLayer( 'relu' );
    net.addLayer( 'tanh' );
    net.addLayer( 'fc', {'num_neurons': 10} );
    net.addLayer( 'softmax' );
    net.print();

    return net;
}

// use mt19937 to pseudo-randomly initialize weights, in a repeatable
// way.  the initialization might not be ideal, but at least it is:
// - random
// - repeatable
// - approximately plausible weights, wont saturate (hopefully) etc
function setWeights( net ) {
    console.log('setting weights...');
    for( var layerId = 0; layerId < net.layers.length; layerId++ ) {
        var layer = net.layers[layerId];
        var layer_type = layer.layer_type;
        if( layer_type != 'conv' && layer_type != 'fc' ) {
            continue;
        }
        console.log('   processing layer id ' + layerId );
        if( layer_type == 'fc' && net.layers[layerId-1].layer_type == 'conv' ) {
            var prev = net.layers[layerId-1];
            console.log('   prev type ' + prev.layer_type );
            mt.seed(0);
            var prev_depth = prev.out_depth;
//            console.log('   prev out depth ' + prev_depth );
//            var prev_filter_length = prev.filters[0].w.length;
            console.log( prev.out_depth + ',' + prev.out_sy + ',' + prev.out_sx );
            var prev_out_plane_size = prev.out_sx * prev.out_sy;
            console.log('   prev out depth ' + prev_depth + ' out planesize ' + prev_out_plane_size );
            // so we need to distribute our random numbers across these input planes
            for( var filterId = 0; filterId < layer.filters.length; filterId++ ) {
                console.log('filter ' + filterId );
                var filter = layer.filters[filterId];
                for( var d = 0; d < prev_depth; d++ ) {
                    for( var xy = 0; xy < prev_out_plane_size; xy++ ) {
                        var thisrand = ( mt() % 100000 ) / 1000000.0;
                        filter.w[ xy * prev_depth + d ] = thisrand;
                    }
                }
            }
            mt.seed(0);
            for( var filterId = 0; filterId < layer.filters.length; filterId++ ) {
                layer.biases.w[filterId] = ( mt() % 100000 ) / 1000000.0;
            }
        } else {
            mt.seed(0);
            for( var filterId = 0; filterId < layer.filters.length; filterId++ ) {
                console.log('filter ' + filterId );
                var filter = layer.filters[filterId];
                for( var d = 0; d < filter.depth; d++ ) {
                    for( var y = 0; y < filter.sy; y++ ) {
                        for( var x = 0; x < filter.sx; x++ ) {
                            filter.set( x, y, d, ( mt() % 100000 ) / 1000000.0 );
                        }
                    }
                }
            }            
            mt.seed(0);
            for( var filterId = 0; filterId < layer.filters.length; filterId++ ) {
                layer.biases.w[filterId] = ( mt() % 100000 ) / 1000000.0;
            }
        }
    }
}

function sampleWeights( net ) {
    for( var layerId = 0; layerId < net.layers.length; layerId++ ) {
        var layer = net.layers[layerId];
        var layerType = layer.layer_type;
        if( layerType != 'conv' && layerType != 'fc' ) {
            continue;
        }
        console.log('   samples weights layer id ' + layerId );
        mt.seed(0);
        var filters = layer.filters;
        var afilter = filters[0];
        for( var i = 0; i < 10; i++ ) { // sample 10 points, pseudo-randomly (but repeatably...)
            var seq = Math.abs( mt() ) % ( filters.length * afilter.depth * afilter.sx * afilter.sy );
            var xysize = afilter.sx * afilter.sy;
            var inoutplane = Math.floor( seq / xysize );
            var outplane = Math.floor( inoutplane / afilter.depth );
            var inplane = inoutplane % afilter.depth;
            var xy = seq % xysize;
            var y = Math.floor( xy / afilter.sx );
            var x = xy % afilter.sx;
            console.log('filter.w[' + outplane + ',' + inplane + ',' + y + ',' + x + ']=' + filters[outplane].get(x,y,inplane).toPrecision(6) );
        }
    }
}

function printForward( net ) {
    console.log('foward results:' );
    for( var layerId = 0; layerId < net.layers.length; layerId++ ) {
        var layer = net.layers[layerId];
        if( layer.layer_type != 'conv' && layer.layer_type != 'fc' ) {
        //    continue;
        }
        console.log('  layer id ' + layerId + ':' );
        var out = layer.out_act;
        mt.seed(0);
        for( var i = 0; i < 10; i++ ) { // sample 10 points, pseudo-randomly (but repeatably...)
            var seq = Math.abs( mt() ) % ( out.sx * out.sy * out.depth );
            var xysize = out.sx * out.sy;
            var d = Math.floor( seq / xysize );
            var xy = seq % xysize;
            var y = Math.floor( xy / out.sx );
            var x = xy % out.sx;
            console.log('out_act.w[' + d + ',' + y + ',' + x + ']=' + layer.out_act.get(x,y,d).toPrecision(6) );
        }
    }
}

function printBackward( net ) {
    console.log('backprop results:' );
    for( var layerId = net.layers.length - 1; layerId > 0; layerId-- ) {
        var layer = net.layers[layerId];
        if( layer.layer_type != 'conv' && layer.layer_type != 'fc' ) {
            continue;
        }
        console.log('   layer id ' + layerId );
        var filter = layer.filters[0];
        var filter_size = filter.sx * filter.sy * filter.depth;
        for( var i = 0; i < filter_size; i++ ) {
            console.log('w[' + i + ']=' + (0.0 + filter.w[i]).toPrecision(6) );
        }
        for( var i = 0; i < 3; i++ ) {
            console.log('bias[' + i + ']=' + (0.0 + layer.biases.w[i]).toPrecision(6) );
        }
    }
}

function learn(options) {
    var labelscontents = fs.readFileSync( __dirname + '/data/mnist_labels.js', { encoding: 'utf-8'} );
    labelscontents = labelscontents.split('=')[1].split(';')[0];
    var labels = JSON.parse(labelscontents);
    console.log('labels.length: ' + labels.length);
    openPng( __dirname + '/data/mnist_batch_0.png', function( data ) {
        var net = createNet();

        setWeights( net );
//        return;
        sampleWeights( net );

        var trainer = new convnetjs.SGDTrainer(net, {method:'sgd', batch_size:options.numTrain, l2_decay:0.00, momentum: 0, learning_rate: 0.4});

        var x = new convnetjs.Vol(28,28,1,0.0);
        for( var it = 0; it < options.numEpochs; it++ ) {
            var numRight = 0;
            var totalLoss = 0;
            for( var i = 0; i < options.numTrain; i++ ) {
                var y = labels[i];
                for( var j = 0; j < 784; j++ ) {
                    var thispoint = data[(i*784+j)*4];
    //                x.w[j] = thispoint/255.0;
    //                x.w[j] = (thispoint-32.7936)*0.00643144;
                     x.w[j] = (thispoint-35.1084)*0.00627357; // this has to match whatever normalization we
                                                              // use on the clconvolve side
                }
                var stats = trainer.train( x, y );
                totalLoss += stats.cost_loss;
                var yhat = net.getPrediction();
                var train_acc = yhat == y ? 1.0 : 0.0;
                numRight += train_acc;
            }
            printForward( net );
            printBackward( net );
            sampleWeights( net );
            var accuracy = numRight * 100.0 / options.numTrain;
            console.log( 'loss ' + totalLoss );
            console.log( 'it ' + it + ' numRight ' + numRight + '/' + options.numTrain +  ' ' + accuracy + '%' );
        }
    });
}

function processArgs(callback) {
    var options = {};
    options.numTrain = 1;
    options.numEpochs = 1;
    for( var i = 2; i < process.argv.length; i++ ) {
        var splitKeyValue = process.argv[i].split('=');
        if( splitKeyValue.length != 2 ) {
            console.log('Please give options as key=value pairs [somekey]=[somevalue]');
            return;
        }
        var key = splitKeyValue[0];
        var value = splitKeyValue[1];
        if( key == 'numtrain' ) {
            options.numTrain = parseInt(value);
        } else if( key == 'numepochs' ) {
            options.numEpochs = parseInt(value);
        } else {
            console.log('key ' + key + ' not recognized. Available keys: numtrain, numepochs');
            return;
        }
    }
    callback(options);
}

processArgs( function(options) {
    doDownloads( function() {
        learn( options );
    });
});


