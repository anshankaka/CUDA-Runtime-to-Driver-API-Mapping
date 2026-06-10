#include "..\..\common\cupti_trace_helper.h"

#include <string.h>

int main()
{
    CUpti_SubscriberHandle subscriber = StartCuptiTrace();

    CUDA_CALL(cudaSetDevice(0));
    cudaStream_t stream = NULL;
    CUDA_CALL(cudaStreamCreate(&stream));

    const int width = 32;
    const int height = 16;
    const size_t bytes = width * height * sizeof(float);
    float *host = (float *)malloc(bytes);
    float *hostOut = (float *)malloc(bytes);
    for (int i = 0; i < width * height; ++i)
    {
        host[i] = (float)i;
        hostOut[i] = 0.0f;
    }

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaArray_t arrayA = NULL;
    cudaArray_t arrayB = NULL;
    cudaArray_t volumeArray = NULL;
    cudaMipmappedArray_t mipmapped = NULL;

    CUDA_CALL(cudaMallocArray(&arrayA, &desc, width, height, cudaArraySurfaceLoadStore));
    CUDA_CALL(cudaMallocArray(&arrayB, &desc, width, height, cudaArraySurfaceLoadStore));
    CUDA_CALL(cudaMemcpy2DToArrayAsync(arrayA, 0, 0, host, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMemcpy2DArrayToArray(arrayB, 0, 0, arrayA, 0, 0, width * sizeof(float), height, cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy2DFromArrayAsync(hostOut, width * sizeof(float), arrayB, 0, 0, width * sizeof(float), height, cudaMemcpyDeviceToHost, stream));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arrayA;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaResourceViewDesc viewDesc;
    memset(&viewDesc, 0, sizeof(viewDesc));
    viewDesc.format = cudaResViewFormatFloat1;
    viewDesc.width = width;
    viewDesc.height = height;
    viewDesc.depth = 0;

    cudaTextureObject_t texture = 0;
    CUDA_CALL(cudaCreateTextureObject(&texture, &resDesc, &texDesc, &viewDesc));
    CUDA_CALL(cudaGetTextureObjectResourceDesc(&resDesc, texture));
    CUDA_CALL(cudaGetTextureObjectTextureDesc(&texDesc, texture));
    CUDA_CALL(cudaGetTextureObjectResourceViewDesc(&viewDesc, texture));

    cudaSurfaceObject_t surface = 0;
    CUDA_CALL(cudaCreateSurfaceObject(&surface, &resDesc));
    CUDA_CALL(cudaGetSurfaceObjectResourceDesc(&resDesc, surface));
    CUDA_CALL(cudaDestroySurfaceObject(surface));
    CUDA_CALL(cudaDestroyTextureObject(texture));

    cudaExtent volumeExtent = make_cudaExtent(width * sizeof(float), height, 4);
    CUDA_CALL(cudaMalloc3DArray(&volumeArray, &desc, volumeExtent));
    cudaMemcpy3DParms copy3d;
    memset(&copy3d, 0, sizeof(copy3d));
    copy3d.srcPtr = make_cudaPitchedPtr(host, width * sizeof(float), width, height);
    copy3d.dstArray = volumeArray;
    copy3d.extent = volumeExtent;
    copy3d.kind = cudaMemcpyHostToDevice;
    CUDA_TRY(cudaMemcpy3DAsync(&copy3d, stream));

    cudaArrayMemoryRequirements arrayReq;
    cudaArraySparseProperties sparseProps;
    memset(&arrayReq, 0, sizeof(arrayReq));
    memset(&sparseProps, 0, sizeof(sparseProps));
    CUDA_TRY(cudaArrayGetMemoryRequirements(&arrayReq, arrayA, 0));
    CUDA_TRY(cudaArrayGetSparseProperties(&sparseProps, arrayA));
    cudaArray_t plane = NULL;
    CUDA_TRY(cudaArrayGetPlane(&plane, arrayA, 0));

    CUDA_CALL(cudaMallocMipmappedArray(&mipmapped, &desc, make_cudaExtent(width, height, 0), 3));
    cudaArray_t level0 = NULL;
    CUDA_CALL(cudaGetMipmappedArrayLevel(&level0, mipmapped, 0));
    memset(&arrayReq, 0, sizeof(arrayReq));
    memset(&sparseProps, 0, sizeof(sparseProps));
    CUDA_TRY(cudaMipmappedArrayGetMemoryRequirements(&arrayReq, mipmapped, 0));
    CUDA_TRY(cudaMipmappedArrayGetSparseProperties(&sparseProps, mipmapped));

    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaFreeMipmappedArray(mipmapped));
    CUDA_CALL(cudaFreeArray(volumeArray));
    CUDA_CALL(cudaFreeArray(arrayB));
    CUDA_CALL(cudaFreeArray(arrayA));
    CUDA_CALL(cudaStreamDestroy(stream));
    free(host);
    free(hostOut);

    StopCuptiTrace(subscriber, "texture surface coverage finished successfully");
    return 0;
}
