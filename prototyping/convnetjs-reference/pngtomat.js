var fs = require('fs');
var convnetjs = require('convnetjs');
var PNG = require('pngjs').PNG;
var labelscontents = fs.readFileSync( __dirname + '/data/mnist_labels.js', { encoding: 'utf-8'} );

eval(labelscontents);
console.log(labels.length);

var setName = 'mnist12k';

function openPng( path, callback ) {
    fs.createReadStream(path)
        .pipe(new PNG({
            filterType: 4
        }))
        .on('parsed', function() {
        console.log('parsed');
        console.log(this.width);
        console.log(this.height);
        callback( this.data );
    });
}

openPng( __dirname + '/data/mnist_batch_0.png', function( data ) {
    console.log( data.length );
    var numImages = data.length / 28 / 28 / 4;
  //  numImages = 128;
    console.log('num images ' + numImages );
//    fs.writeFileSync( __dirname + "/train.idx'
    var labelsBuffer = new Buffer( numImages * 4 + 5 * 4 );
    labelsBuffer.writeIntLE( 0x1E3D4C54, 0, 4 );
    console.log( labelsBuffer.readIntLE( 0, 1 ) );
    labelsBuffer.writeIntLE( 1, 1 * 4, 4 );
    labelsBuffer.writeIntLE( numImages, 2 * 4, 4 ); 
    labelsBuffer.writeIntLE( 1, 3 * 4, 4 ); 
    labelsBuffer.writeIntLE( 1, 4 * 4, 4 ); 
    for( var i = 0; i < 24; i++ ) {
        console.log( i + ' ' + labelsBuffer.readUIntLE( i, 1 ) );
    }
    for( var i = 0; i < numImages; i++ ) {
        labelsBuffer.writeIntLE( labels[i], ( 5 + i ) * 4, 4 );
    }
    fs.writeFileSync( __dirname + '/data/' + setName + '-cat.mat', labelsBuffer );

    imagesBuffer = new Buffer( numImages * 28 * 28 + 6 * 4 );
    imagesBuffer.writeIntLE( 0x1e3d4c55, 0 * 4, 4 );
    imagesBuffer.writeIntLE( 4, 1 * 4, 4 );
    imagesBuffer.writeIntLE( numImages, 2 * 4, 4 );
    imagesBuffer.writeIntLE( 1, 3 * 4, 4 );
    imagesBuffer.writeIntLE( 28, 4 * 4, 4 );
    imagesBuffer.writeIntLE( 28, 5 * 4, 4 );
    for( var i = 0; i < numImages * 28 * 28; i++ ) {
//        console.log( i + ' ' + data[i] );
        imagesBuffer.writeUIntLE( data[i*4], 6 * 4 + i, 1 );
    }
    fs.writeFileSync( __dirname + '/data/' + setName + '-dat.mat', imagesBuffer );
});


