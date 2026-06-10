#include "..\..\common\cupti_trace_helper.h"

#include <string.h>

__device__ int gDeviceSymbol[64];

int main()
{
    CUpti_SubscriberHandle subscriber = StartCuptiTrace();

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CALL(cudaSetDevice(device));
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));

    const size_t count = 1024;
    const size_t bytes = count * sizeof(int);
    int *host = (int *)malloc(bytes);
    int *hostOut = (int *)malloc(bytes);
    for (size_t i = 0; i < count; ++i)
    {
        host[i] = (int)i;
        hostOut[i] = 0;
    }

    int *dev = NULL;
    int *managed = NULL;
    int *registered = (int *)malloc(bytes);
    memset(registered, 7, bytes);

    CUDA_CALL(cudaMalloc((void **)&dev, bytes));
    CUDA_CALL(cudaMallocManaged((void **)&managed, bytes));
    CUDA_CALL(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(hostOut, dev, bytes, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpyToSymbol(gDeviceSymbol, host, 64 * sizeof(int)));
    CUDA_CALL(cudaMemcpyFromSymbol(hostOut, gDeviceSymbol, 64 * sizeof(int)));
    void *symbolPtr = NULL;
    size_t symbolSize = 0;
    CUDA_CALL(cudaGetSymbolAddress(&symbolPtr, gDeviceSymbol));
    CUDA_CALL(cudaGetSymbolSize(&symbolSize, gDeviceSymbol));

    CUDA_CALL(cudaMemset(dev, 1, count * sizeof(int)));
    CUDA_TRY(cudaMemAdvise(managed, count * sizeof(int), cudaMemAdviseSetPreferredLocation, device));
    CUDA_TRY(cudaMemPrefetchAsync(managed, count * sizeof(int), device));
    CUDA_CALL(cudaDeviceSynchronize());

    int location = 0;
    CUDA_TRY(cudaMemRangeGetAttribute(&location, sizeof(location), cudaMemRangeAttributePreferredLocation, managed, count * sizeof(int)));
    void *attrs[] = {&location};
    size_t attrSizes[] = {sizeof(location)};
    cudaMemRangeAttribute attrKinds[] = {cudaMemRangeAttributePreferredLocation};
    CUDA_TRY(cudaMemRangeGetAttributes(attrs, attrSizes, attrKinds, 1, managed, count * sizeof(int)));

    size_t freeMem = 0;
    size_t totalMem = 0;
    CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));

    cudaError_t registerResult = CUDA_TRY(cudaHostRegister(registered, bytes, cudaHostRegisterDefault));
    if (registerResult == cudaSuccess)
    {
        int *registeredDev = NULL;
        CUDA_TRY(cudaHostGetDevicePointer((void **)&registeredDev, registered, 0));
        unsigned int hostFlags = 0;
        CUDA_TRY(cudaHostGetFlags(&hostFlags, registered));
        CUDA_TRY(cudaHostUnregister(registered));
    }

    int *pitchDev = NULL;
    size_t pitch = 0;
    const size_t widthBytes = 64 * sizeof(int);
    const size_t height = 16;
    CUDA_CALL(cudaMallocPitch((void **)&pitchDev, &pitch, widthBytes, height));
    CUDA_CALL(cudaMemcpy2D(pitchDev, pitch, host, widthBytes, widthBytes, height, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset2D(pitchDev, pitch, 0, widthBytes, height));

    cudaExtent extent = make_cudaExtent(widthBytes, height, 4);
    cudaPitchedPtr pitched;
    CUDA_CALL(cudaMalloc3D(&pitched, extent));
    cudaMemcpy3DParms copy3d = {0};
    copy3d.srcPtr = make_cudaPitchedPtr(host, widthBytes, 64, height);
    copy3d.dstPtr = pitched;
    copy3d.extent = extent;
    copy3d.kind = cudaMemcpyHostToDevice;
    CUDA_CALL(cudaMemcpy3D(&copy3d));
    CUDA_CALL(cudaMemset3D(pitched, 0, extent));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaArray_t array2d = NULL;
    CUDA_CALL(cudaMallocArray(&array2d, &desc, 64, height));
    CUDA_CALL(cudaMemcpy2DToArray(array2d, 0, 0, host, widthBytes, widthBytes, height, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy2DFromArray(hostOut, widthBytes, array2d, 0, 0, widthBytes, height, cudaMemcpyDeviceToHost));
    cudaChannelFormatDesc outDesc;
    cudaExtent outExtent;
    unsigned int outFlags = 0;
    CUDA_CALL(cudaArrayGetInfo(&outDesc, &outExtent, &outFlags, array2d));

    CUDA_CALL(cudaFreeArray(array2d));
    CUDA_CALL(cudaFree(pitchDev));
    CUDA_CALL(cudaFree(pitched.ptr));
    CUDA_CALL(cudaFree(managed));
    CUDA_CALL(cudaFree(dev));
    free(host);
    free(hostOut);
    free(registered);

    CUDA_CALL(cudaDeviceSynchronize());
    StopCuptiTrace(subscriber, "memory coverage finished successfully");
    return 0;
}
