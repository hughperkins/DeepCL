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
#include "trainers/Adadelta.h"

#include "weights/UniformInitializer.h"
#include "weights/OriginalInitializer.h"

#include "normalize/NormalizationHelper.h"
#include "layer/Layer.h"
#include "conv/ConvolutionalLayer.h"
#include "input/InputLayer.h"
#include "layer/LayerMakers.h"

#include "batch/BatchProcess.h"
#include "batch/NetLearner.h"
#include "batch/NetLearnerOnDemand.h"
#include "batch/NetLearnerOnDemandv2.h"

#include "weights/WeightsPersister.h"
#include "util/FileHelper.h"
#include "loaders/GenericLoader.h"
#include "loaders/GenericLoaderv2.h"

#include "clblas/ClBlasInstance.h"

#include "DeepCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class DeepCL_EXPORT DeepCL : public EasyCL {
public:
//    EasyCL *cl;
    ClBlasInstance clBlasInstance;
    
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    DeepCL(cl_platform_id platformId, cl_device_id deviceId);
    ~DeepCL();
    void deleteMe();
    STATIC DeepCL *createForFirstGpu();
    STATIC DeepCL *createForFirstGpuOtherwiseCpu();
    STATIC DeepCL *createForIndexedDevice(int device);
    STATIC DeepCL *createForIndexedGpu(int gpu);
    STATIC DeepCL *createForPlatformDeviceIndexes(int platformIndex, int deviceIndex);
    STATIC DeepCL *createForPlatformDeviceIds(cl_platform_id platformId, cl_device_id deviceId);

    // [[[end]]]
};

