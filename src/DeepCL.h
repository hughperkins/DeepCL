#pragma once

// convenience header, to include what we need, without causing whole world to rebuild
// at the same time :-) (cf, if we put in NeuralNet.h)

#include "net/NeuralNet.h"
#include "trainers/SGD.h"
#include "EasyCL.h"
#include "trainers/Trainer.h"
#include "layer/Layer.h"
//#include "batch/EpochMaker.h"
#include "conv/ConvolutionalLayer.h"
#include "input/InputLayer.h"
#include "net/Trainable.h"
//#include "NeuralNetMould.h"
#include "input/InputLayerMaker.h"
#include "conv/ConvolutionalMaker.h"
#include "patches/RandomTranslationsMaker.h"
#include "patches/RandomPatchesMaker.h"
#include "normalize/NormalizationLayerMaker.h"
#include "fc/FullyConnectedMaker.h"
#include "loaders/GenericLoader.h"
#include "normalize/NormalizationHelper.h"
#include "batch/BatchProcess.h"
#include "netdef/NetdefToNet.h"
#include "weights/WeightsPersister.h"
#include "util/FileHelper.h"
#include "net/MultiNet.h"
#include "batch/NetLearner.h"
#include "batch/NetLearnerOnDemand.h"
#include "layer/LayerMakers.h"

