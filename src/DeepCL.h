#pragma once

// convenience header, to include what we need, without causing whole world to rebuild
// at the same time :-) (cf, if we put in NeuralNet.h)

#include "EasyCL.h"

#include "netdef/NetdefToNet.h"
#include "net/Trainable.h"
#include "net/NeuralNet.h"
#include "net/MultiNet.h"

#include "trainers/Trainer.h"
#include "trainers/SGD.h"
#include "trainers/Annealer.h"
#include "trainers/Nesterov.h"
#include "trainers/Adagrad.h"
#include "trainers/Rmsprop.h"

#include "normalize/NormalizationHelper.h"
#include "layer/Layer.h"
#include "conv/ConvolutionalLayer.h"
#include "input/InputLayer.h"
#include "layer/LayerMakers.h"

#include "batch/BatchProcess.h"
#include "batch/NetLearner.h"
#include "batch/NetLearnerOnDemand.h"

#include "weights/WeightsPersister.h"
#include "util/FileHelper.h"
#include "loaders/GenericLoader.h"

