#include "..\..\common\cupti_trace_helper.h"

#include <string.h>

__global__ void DeviceExecKernel(double *out, double value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        out[0] = value * 2.0;
    }
}

static size_t DynamicSmemSize(int blockSize)
{
    return (size_t)(blockSize / 32) * sizeof(int);
}

int main()
{
    CUpti_SubscriberHandle subscriber = StartCuptiTrace();

    int device = 0;
    int count = 0;
    int chosen = 0;
    int value = 0;
    unsigned int flags = 0;
    int driverVersion = 0;
    int runtimeVersion = 0;
    size_t limit = 0;
    size_t textureWidth = 0;
    cudaDeviceProp prop;
    cudaFuncAttributes attrs;

    CUDA_TRY(cudaSetDeviceFlags(cudaDeviceScheduleAuto));
    CUDA_CALL(cudaGetDeviceCount(&count));
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));
    CUDA_TRY(cudaChooseDevice(&chosen, &prop));
    CUDA_TRY(cudaInitDevice(device, 0, 0));
    CUDA_TRY(cudaSetValidDevices(&device, 1));
    CUDA_TRY(cudaGetDeviceFlags(&flags));
    CUDA_CALL(cudaDriverGetVersion(&driverVersion));
    CUDA_CALL(cudaRuntimeGetVersion(&runtimeVersion));

    CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, device));
    char pciBusId[32] = {0};
    CUDA_CALL(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device));
    CUDA_CALL(cudaDeviceGetByPCIBusId(&chosen, pciBusId));
    CUDA_CALL(cudaDeviceGetCacheConfig((cudaFuncCache *)&value));
    CUDA_TRY(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    CUDA_CALL(cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize));
    CUDA_TRY(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, limit));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    CUDA_TRY(cudaDeviceGetTexture1DLinearMaxWidth(&textureWidth, &desc, device));
    if (count > 1)
    {
        CUDA_TRY(cudaDeviceGetP2PAttribute(&value, cudaDevP2PAttrPerformanceRank, 0, 1));
    }

    cudaMemPool_t pool = NULL;
    CUDA_TRY(cudaDeviceGetDefaultMemPool(&pool, device));
    if (pool != NULL)
    {
        CUDA_TRY(cudaDeviceSetMemPool(device, pool));
    }
    CUDA_TRY(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice, cudaFlushGPUDirectRDMAWritesToOwner));

    CUDA_CALL(cudaFuncGetAttributes(&attrs, (const void *)DeviceExecKernel));
    const char *kernelName = NULL;
    CUDA_CALL(cudaFuncGetName(&kernelName, (const void *)DeviceExecKernel));
    size_t paramOffset = 0;
    size_t paramSize = 0;
    CUDA_TRY(cudaFuncGetParamInfo((const void *)DeviceExecKernel, 0, &paramOffset, &paramSize));
    CUDA_TRY(cudaFuncSetCacheConfig((const void *)DeviceExecKernel, cudaFuncCachePreferShared));
    CUDA_TRY(cudaFuncSetAttribute((const void *)DeviceExecKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50));

    int blocksPerSm = 0;
    int minGridSize = 0;
    int blockSize = 0;
    int clusterSize = 0;
    size_t availableSmem = 0;
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, (const void *)DeviceExecKernel, 64, 0));
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&blocksPerSm, (const void *)DeviceExecKernel, 64, 0, cudaOccupancyDefault));
    CUDA_CALL(cudaOccupancyAvailableDynamicSMemPerBlock(&availableSmem, (const void *)DeviceExecKernel, blocksPerSm, 64));
    CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, DeviceExecKernel, 0, 0));
    CUDA_CALL(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, DeviceExecKernel, DynamicSmemSize, 0));

    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = dim3(1);
    config.blockDim = dim3(64);
    config.dynamicSmemBytes = 0;
    config.stream = 0;
    config.attrs = NULL;
    config.numAttrs = 0;
    CUDA_TRY(cudaOccupancyMaxPotentialClusterSize(&clusterSize, (const void *)DeviceExecKernel, &config));
    CUDA_TRY(cudaOccupancyMaxActiveClusters(&clusterSize, (const void *)DeviceExecKernel, &config));

    double *out = NULL;
    double valueArg = 3.0;
    void *args[] = {&out, &valueArg};
    CUDA_CALL(cudaMalloc((void **)&out, sizeof(double)));
    CUDA_CALL(cudaLaunchKernelExC(&config, (const void *)DeviceExecKernel, args));
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaFree(out));

    CUDA_CALL(cudaGetLastError());
    cudaError_t fake = cudaErrorInvalidValue;
    (void)cudaGetErrorName(fake);
    (void)cudaGetErrorString(fake);

    StopCuptiTrace(subscriber, "device exec coverage finished successfully");
    return 0;
}
