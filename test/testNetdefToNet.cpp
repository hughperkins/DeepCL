// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

#include "NeuralNet.h"
#include "NetdefToNet.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "SoftMaxLayer.h"

TEST( testNetdefToNet, empty ) {
    NeuralNet *net = new NeuralNet();
    net->addLayer( InputLayerMaker<unsigned char>::instance()->numPlanes(1)->boardSize(19) );
    EXPECT_EQ( true, NetdefToNet::createNetFromNetdef( net, "" ) );
    EXPECT_EQ( 2, net->layers.size() );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->layers[1] ) != 0 );
    delete net;
}

TEST( testNetdefToNet, onefc ) {
    NeuralNet *net = new NeuralNet();
    net->addLayer( InputLayerMaker<unsigned char>::instance()->numPlanes(1)->boardSize(19) );
    EXPECT_EQ( true, NetdefToNet::createNetFromNetdef( net, "150n" ) );
    EXPECT_EQ( 3, net->layers.size() );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->layers[1] ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->layers[2] ) != 0 );
    FullyConnectedLayer *fc = dynamic_cast< FullyConnectedLayer * >( net->layers[1] );
    EXPECT_EQ( 1, fc->boardSize );
    EXPECT_EQ( 150, fc->numPlanes );
    EXPECT_EQ( 150, fc->numPlanes );
    EXPECT_EQ( "LINEAR", fc->fn->getDefineName() );
    delete net;
}

TEST( testNetdefToNet, onefclinear ) {
    NeuralNet *net = new NeuralNet();
    net->addLayer( InputLayerMaker<unsigned char>::instance()->numPlanes(1)->boardSize(19) );
    EXPECT_EQ( true, NetdefToNet::createNetFromNetdef( net, "150n" ) );
    EXPECT_EQ( 3, net->layers.size() );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->layers[1] ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->layers[2] ) != 0 );
    FullyConnectedLayer *fc = dynamic_cast< FullyConnectedLayer * >( net->layers[1] );
    EXPECT_EQ( 1, fc->boardSize );
    EXPECT_EQ( 150, fc->numPlanes );
    EXPECT_EQ( 150, fc->numPlanes );
    EXPECT_EQ( "LINEAR", fc->fn->getDefineName() );
    delete net;
}

TEST( testNetdefToNet, DISABLED_3xfclinear ) {
    NeuralNet *net = new NeuralNet();
    net->addLayer( InputLayerMaker<unsigned char>::instance()->numPlanes(1)->boardSize(19) );
    EXPECT_EQ( true, NetdefToNet::createNetFromNetdef( net, "3*150n{linear}" ) );
    net->print();
    EXPECT_EQ( 5, net->layers.size() );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->layers[1] ) != 0 );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->layers[2] ) != 0 );
    EXPECT_TRUE( dynamic_cast< FullyConnectedLayer * >( net->layers[3] ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->layers[4] ) != 0 );
    FullyConnectedLayer *fc = dynamic_cast< FullyConnectedLayer * >( net->layers[1] );
    EXPECT_EQ( 1, fc->boardSize );
    EXPECT_EQ( 150, fc->numPlanes );
    EXPECT_EQ( "LINEAR", fc->fn->getDefineName() );
    delete net;
}

TEST( testNetdefToNet, DISABLED_3x32c5z ) {
    NeuralNet *net = new NeuralNet();
    net->addLayer( InputLayerMaker<unsigned char>::instance()->numPlanes(1)->boardSize(19) );
    EXPECT_EQ( true, NetdefToNet::createNetFromNetdef( net, "3*32c5{z}" ) );
    net->print();
    EXPECT_EQ( 5, net->layers.size() );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->layers[1] ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->layers[2] ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->layers[3] ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->layers[4] ) != 0 );
    ConvolutionalLayer *conv = dynamic_cast< ConvolutionalLayer * >( net->layers[1] );
    EXPECT_EQ( 19, conv->dim.inputBoardSize );
    EXPECT_EQ( true, conv->dim.padZeros );
    EXPECT_EQ( 1, conv->dim.inputPlanes );
    EXPECT_EQ( 32, conv->dim.numFilters );
    EXPECT_EQ( "RELU", conv->activationFunction->getDefineName() );
    delete net;
}

TEST( testNetdefToNet, DISABLED_2x32c7_3x32c5z ) {
    NeuralNet *net = new NeuralNet();
    net->addLayer( InputLayerMaker<unsigned char>::instance()->numPlanes(1)->boardSize(19) );
    EXPECT_EQ( true, NetdefToNet::createNetFromNetdef( net, "2*32c7{z}-3*32c5{z}" ) );
    net->print();
    EXPECT_EQ( 7, net->layers.size() );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->layers[1] ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->layers[2] ) != 0 );
    EXPECT_TRUE( dynamic_cast< ConvolutionalLayer * >( net->layers[3] ) != 0 );
    EXPECT_TRUE( dynamic_cast< SoftMaxLayer * >( net->layers[6] ) != 0 );
    ConvolutionalLayer *conv = dynamic_cast< ConvolutionalLayer * >( net->layers[1] );
    EXPECT_EQ( 19, conv->dim.inputBoardSize );
    EXPECT_EQ( true, conv->dim.padZeros );
    EXPECT_EQ( 1, conv->dim.inputPlanes );
    EXPECT_EQ( 32, conv->dim.numFilters );
    EXPECT_EQ( "RELU", conv->activationFunction->getDefineName() );
    delete net;
}

