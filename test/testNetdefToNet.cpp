// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

#include "net/NeuralNet.h"
#include "netdef/NetdefToNet.h"
#include "pooling/PoolingLayer.h"
#include "fc/FullyConnectedLayer.h"
#include "conv/ConvolutionalLayer.h"
#include "loss/SoftMaxLayer.h"
#include "layer/LayerMakers.h"

TEST( testNetdefToNet, empty ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = new NeuralNet(cl);
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(19) );
    EXPECT_EQ( true, NetdefToNet::createNetFromNetdef( net, "" ) );
    EXPECT_EQ( 2, net->getNumLayers() );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->getLayer(1) ) != 0 );
    delete net;
    delete cl;
}

TEST( testNetdefToNet, onefc ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = new NeuralNet(cl);
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(19) );
    EXPECT_EQ( true, NetdefToNet::createNetFromNetdef( net, "150n" ) );
    EXPECT_EQ( 3, net->getNumLayers() );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->getLayer(1) ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->getLayer(2) ) != 0 );
    FullyConnectedLayer *fc = dynamic_cast< FullyConnectedLayer * >( net->getLayer(1) );
    EXPECT_EQ( 1, fc->imageSize );
    EXPECT_EQ( 150, fc->numPlanes );
    EXPECT_EQ( 150, fc->numPlanes );
//    EXPECT_EQ( "LINEAR", fc->fn->getDefineName() );
    delete net;
    delete cl;
}

TEST( testNetdefToNet, onefclinear ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = new NeuralNet(cl);
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(19) );
    EXPECT_EQ( true, NetdefToNet::createNetFromNetdef( net, "150n" ) );
    EXPECT_EQ( 3, net->getNumLayers() );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->getLayer(1) ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->getLayer(2) ) != 0 );
    FullyConnectedLayer *fc = dynamic_cast< FullyConnectedLayer * >( net->getLayer(1) );
    EXPECT_EQ( 1, fc->imageSize );
    EXPECT_EQ( 150, fc->numPlanes );
    EXPECT_EQ( 150, fc->numPlanes );
//    EXPECT_EQ( "LINEAR", fc->fn->getDefineName() );
    delete net;
    delete cl;
}

TEST( testNetdefToNet, 150n_10n ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = new NeuralNet(cl);
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(19) );
    EXPECT_EQ( true, NetdefToNet::createNetFromNetdef( net, "150n-10n" ) );
    EXPECT_EQ( 4, net->getNumLayers() );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->getLayer(1) ) != 0 );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->getLayer(2) ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->getLayer(3) ) != 0 );
    FullyConnectedLayer *fc1 = dynamic_cast< FullyConnectedLayer * >( net->getLayer(1) );
    EXPECT_EQ( 1, fc1->imageSize );
    EXPECT_EQ( 150, fc1->numPlanes );
//    EXPECT_EQ( "TANH", fc1->fn->getDefineName() );

    FullyConnectedLayer *fc2 = dynamic_cast< FullyConnectedLayer * >( net->getLayer(2) );
    EXPECT_EQ( 1, fc2->imageSize );
    EXPECT_EQ( 10, fc2->numPlanes );
//    EXPECT_EQ( "LINEAR", fc2->fn->getDefineName() );

    delete net;
    delete cl;
}

TEST( testNetdefToNet, 3xfclinear ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = new NeuralNet(cl);
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(19) );
    ASSERT_EQ( true, NetdefToNet::createNetFromNetdef( net, "3*150n" ) );
    net->print();
    EXPECT_EQ( 5, net->getNumLayers() );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->getLayer(1) ) != 0 );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->getLayer(2) ) != 0 );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->getLayer(3) ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->getLayer(4) ) != 0 );
    FullyConnectedLayer *fc = dynamic_cast< FullyConnectedLayer * >( net->getLayer(1) );
    EXPECT_EQ( 1, fc->imageSize );
    EXPECT_EQ( 150, fc->numPlanes );
//    EXPECT_EQ( "LINEAR", fc->fn->getDefineName() );
    delete net;
    delete cl;
}

TEST( testNetdefToNet, mp2_3x32c5z_10n ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = new NeuralNet(cl);
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(19) );
    ASSERT_EQ( true, NetdefToNet::createNetFromNetdef( net, "mp2-3*32c5z-10n " ) );
    net->print();
    EXPECT_EQ( 7, net->getNumLayers() );
    EXPECT_TRUE( dynamic_cast< PoolingLayer * >( net->getLayer(1) ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->getLayer(2) ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->getLayer(3) ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->getLayer(4) ) != 0 );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->getLayer(5) ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->getLayer(6) ) != 0 );
    FullyConnectedLayer *fc = dynamic_cast< FullyConnectedLayer * >( net->getLayer(5) );
    EXPECT_EQ( 1, fc->imageSize );
    EXPECT_EQ( 10, fc->numPlanes );
//    EXPECT_EQ( "LINEAR", fc->fn->getDefineName() );
    delete net;
    delete cl;
}

TEST( testNetdefToNet, 3x32c5zmp2 ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = new NeuralNet(cl);
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(128) );
    ASSERT_EQ( true, NetdefToNet::createNetFromNetdef( net, "3*(32c5z-mp2)-10n" ) );
    net->print();
    EXPECT_EQ( 9, net->getNumLayers() );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->getLayer(1) ) != 0 );
    EXPECT_TRUE( dynamic_cast< PoolingLayer * >( net->getLayer(2) ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->getLayer(3) ) != 0 );
    EXPECT_TRUE( dynamic_cast< PoolingLayer * >( net->getLayer(4) ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->getLayer(5) ) != 0 );
    EXPECT_TRUE( dynamic_cast< PoolingLayer * >( net->getLayer(6) ) != 0 );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->getLayer(7) ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->getLayer(8) ) != 0 );

    ConvolutionalLayer *conv = dynamic_cast< ConvolutionalLayer * >( net->getLayer(1) );
    EXPECT_EQ( 128, conv->dim.inputSize );
    EXPECT_EQ( true, conv->dim.padZeros );
    EXPECT_EQ( 1, conv->dim.inputPlanes );
    EXPECT_EQ( 32, conv->dim.numFilters );
//    EXPECT_EQ( "RELU", conv->activationFunction->getDefineName() );
    delete net;
    delete cl;
}

TEST( testNetdefToNet, 2x32c7_3x32c5z ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = new NeuralNet(cl);
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(19) );
    EXPECT_EQ( true, NetdefToNet::createNetFromNetdef( net, "2*32c7z-3*32c5z-10n" ) );
    net->print();
    EXPECT_EQ( 8, net->getNumLayers() );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->getLayer(1) ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->getLayer(2) ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->getLayer(3) ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->getLayer(4) ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->getLayer(5) ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->getLayer(7) ) != 0 );
    ConvolutionalLayer *conv = dynamic_cast< ConvolutionalLayer * >( net->getLayer(1) );
    EXPECT_EQ( 19, conv->dim.inputSize );
    EXPECT_EQ( true, conv->dim.padZeros );
    EXPECT_EQ( 1, conv->dim.inputPlanes );
    EXPECT_EQ( 32, conv->dim.numFilters );
    EXPECT_EQ( 7, conv->dim.filterSize );
//    EXPECT_EQ( "RELU", conv->activationFunction->getDefineName() );

    conv = dynamic_cast< ConvolutionalLayer * >( net->getLayer(2) );
    EXPECT_EQ( 19, conv->dim.inputSize );
    EXPECT_EQ( true, conv->dim.padZeros );
    EXPECT_EQ( 32, conv->dim.inputPlanes );
    EXPECT_EQ( 32, conv->dim.numFilters );
    EXPECT_EQ( 7, conv->dim.filterSize );
//    EXPECT_EQ( "RELU", conv->activationFunction->getDefineName() );

    conv = dynamic_cast< ConvolutionalLayer * >( net->getLayer(3) );
    EXPECT_EQ( 19, conv->dim.inputSize );
    EXPECT_EQ( true, conv->dim.padZeros );
    EXPECT_EQ( 32, conv->dim.inputPlanes );
    EXPECT_EQ( 32, conv->dim.numFilters );
    EXPECT_EQ( 5, conv->dim.filterSize );
//    EXPECT_EQ( "RELU", conv->activationFunction->getDefineName() );

    conv = dynamic_cast< ConvolutionalLayer * >( net->getLayer(5) );
    EXPECT_EQ( 19, conv->dim.inputSize );
    EXPECT_EQ( true, conv->dim.padZeros );
    EXPECT_EQ( 32, conv->dim.inputPlanes );
    EXPECT_EQ( 32, conv->dim.numFilters );
    EXPECT_EQ( 5, conv->dim.filterSize );
//    EXPECT_EQ( "RELU", conv->activationFunction->getDefineName() );

    delete net;
    delete cl;
}

