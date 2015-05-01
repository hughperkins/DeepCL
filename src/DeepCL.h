#pragma once

// convenience header, to include what we need, without causing whole world to rebuild
// at the same time :-) (cf, if we put in NeuralNet.h)

#include "NeuralNet.h"
#include "SGD.h"
#include "EasyCL.h"
#include "Trainer.h"
#include "Layer.h"
#include "EpochMaker.h"
#include "ConvolutionalLayer.h"
#include "InputLayer.h"
#include "Trainable.h"
#include "NeuralNetMould.h"
#include "InputLayerMaker.h"
#include "ConvolutionalMaker.h"
#include "RandomTranslationsMaker.h"
#include "RandomPatchesMaker.h"
#include "NormalizationLayerMaker.h"
#include "FullyConnectedMaker.h"
#include "GenericLoader.h"
#include "NormalizationHelper.h"
#include "BatchProcess.h"
#include "NetdefToNet.h"
#include "WeightsPersister.h"
#include "FileHelper.h"
#include "MultiNet.h"
#include "NetLearner.h"
#include "NetLearnerOnDemand.h"
#include "LayerMakers.h"

