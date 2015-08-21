#include "DeepCL.h"
#include "DevicesInfo.h"

#undef STATIC
#define STATIC
#define PUBLIC

using namespace easycl;

//DeepCL::DeepCL() :
//    EasyCL() {
//}
//DeepCL::DeepCL(int gpu) :
//    EasyCL(gpu) {
//}
PUBLIC DeepCL::DeepCL(cl_platform_id platformId, cl_device_id deviceId) :
    EasyCL(platformId, deviceId) {
}
PUBLIC DeepCL::~DeepCL() {
}
PUBLIC void DeepCL::deleteMe() {
    delete this;
}
PUBLIC STATIC DeepCL *DeepCL::createForFirstGpu() {
    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedGpu(0, &platformId, &deviceId);
    return new DeepCL(platformId, deviceId);
}
PUBLIC STATIC DeepCL *DeepCL::createForFirstGpuOtherwiseCpu() {
    if(DevicesInfo::getNumGpus() >= 1) {
        return createForFirstGpu();
    } else {
        return createForIndexedDevice(0);
    }
}
PUBLIC STATIC DeepCL *DeepCL::createForIndexedDevice(int device) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedDevice(device, &platformId, &deviceId);
    return new DeepCL(platformId, deviceId);
}
PUBLIC STATIC DeepCL *DeepCL::createForIndexedGpu(int gpu) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedGpu(gpu, &platformId, &deviceId);
    return new DeepCL(platformId, deviceId);
}
PUBLIC STATIC DeepCL *DeepCL::createForPlatformDeviceIndexes(int platformIndex, int deviceIndex) {
    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedPlatformDevice(platformIndex, deviceIndex, CL_DEVICE_TYPE_ALL, &platformId, &deviceId);
    return new DeepCL(platformId, deviceId);
}
PUBLIC STATIC DeepCL *DeepCL::createForPlatformDeviceIds(cl_platform_id platformId, cl_device_id deviceId) {
    return new DeepCL(platformId, deviceId);
}

